import torch, os, yaml
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Dict

from build.build_classifier import build_classifier
from build.build_encoder import build_encoder
from data.datasets import get_classification_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierTrainer:
    def __init__(self, config_path: str = "build/config.yaml"):
        """
        Initialize the classifier trainer using the modified datasets.py
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Build encoder
        self.encoder = build_encoder(
            encoder_name=self.config['encoder']['name'], # type: ignore
            config_path=config_path, 
            device=device
        )
        self.encoder.eval()  # Set encoder to evaluation mode
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Create dataloaders using your existing dataset structure
        self.train_loader, self.test_loader, self.num_classes, self.class_to_idx = get_classification_loader(
            train_dir=self.config['data']['train_dir'],# type: ignore
            test_dir=self.config['data'].get('test_dir', None),# type: ignore
            batch_size=self.config['training']['batch_size'],# type: ignore
            num_workers=self.config['training'].get('num_workers', 8)# type: ignore
        )
        
        # Update classifier config with num_classes
        if 'classifier' not in self.config:# type: ignore
            self.config['classifier'] = {}# type: ignore
        if 'kwargs' not in self.config['classifier']:# type: ignore
            self.config['classifier']['kwargs'] = {}# type: ignore
        self.config['classifier']['kwargs']['num_classes'] = self.num_classes# type: ignore
        
        # Build classifier
        self.classifier = build_classifier(config_path=config_path, device=device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=self.config['training']['learning_rate'],# type: ignore
            weight_decay=self.config['training'].get('weight_decay', 1e-4)# type: ignore
        )
        
        # Setup scheduler if specified
        if 'scheduler' in self.config['training']:# type: ignore
            scheduler_config = self.config['training']['scheduler']# type: ignore
            if scheduler_config['type'] == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, 
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            else:
                self.scheduler = None# type: ignore
        else:
            self.scheduler = None# type: ignore
    
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)# type: ignore
    
    def extract_features(self, images):
        """Extract features using the encoder."""
        with torch.no_grad():
            features = self.encoder(images)
        return features
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features using encoder
            features = self.extract_features(images)
            
            # Forward pass through classifier
            self.optimizer.zero_grad()
            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                
                # Extract features using encoder
                features = self.extract_features(images)
                
                # Forward pass through classifier
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'class_to_idx': self.class_to_idx
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_dir = self.config['training'].get('checkpoint_dir', 'checkpoints')# type: ignore
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'classifier_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'classifier_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved to {best_path}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']# type: ignore
        best_accuracy = 0.0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Number of classes: {self.num_classes}")
        print(f"Training samples: {len(self.train_loader.dataset)}")# type: ignore
        if self.test_loader:
            print(f"Test samples: {len(self.test_loader.dataset)}")# type: ignore
        print(f"Classes: {list(self.class_to_idx.keys())}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            if self.test_loader:
                test_metrics = self.evaluate(self.test_loader)
                print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.2f}%")
                
                # Check if this is the best model
                is_best = test_metrics['accuracy'] > best_accuracy
                if is_best:
                    best_accuracy = test_metrics['accuracy']
                
                # Save checkpoint
                self.save_checkpoint(epoch, test_metrics, is_best)
            else:
                # Use train accuracy if no test set
                is_best = train_metrics['accuracy'] > best_accuracy
                if is_best:
                    best_accuracy = train_metrics['accuracy']
                
                self.save_checkpoint(epoch, train_metrics, is_best)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Best Accuracy: {best_accuracy:.2f}%")
        
        print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")

def main():
    """Main function to run training."""
    trainer = ClassifierTrainer()
    trainer.train()

if __name__ == "__main__":
    main()