from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from torchvision import transforms
import numpy as np

NUM_DATASET_WORKERS = 8

class GenericImageDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        # collect all jpg/png paths
        self.img_paths = []
        for d in data_dirs:
            self.img_paths += glob(os.path.join(d, '*.jpg'))
            self.img_paths += glob(os.path.join(d, '*.png'))
        self.img_paths.sort()
        # we’ll ignore any passed‐in transform and apply size‐aware cropping below

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = Image.open(image_path).convert('RGB')
        # PIL.Image.size → (width, height)
        w, h = image.size
        # crop each dim down to a multiple of 128
        if h % 128 != 0 or w % 128 != 0:
            h = h - (h % 128)
            w = w - (w % 128)
        # center‐crop to (height, width), then to tensor
        crop_and_to_tensor = transforms.Compose([
            transforms.CenterCrop((h, w)),
            transforms.ToTensor()
        ])
        return crop_and_to_tensor(image)

    def __len__(self):
        return len(self.img_paths)

class ClassificationImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for classification with folder structure: root_dir/<class_name>/*.jpg|*.png
        
        Args:
            root_dir: Root directory containing class folders
            transform: Optional transform (will be ignored, using size-aware cropping)
        """
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Get all class directories
        class_names = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all jpg and png files in this class directory
            jpg_files = glob(os.path.join(class_dir, '*.jpg'))
            png_files = glob(os.path.join(class_dir, '*.png'))
            class_img_paths = jpg_files + png_files
            
            self.img_paths.extend(class_img_paths)
            self.labels.extend([class_idx] * len(class_img_paths)) # type: ignore
        
        # Sort to ensure consistent ordering
        sorted_data = sorted(zip(self.img_paths, self.labels))
        self.img_paths, self.labels = zip(*sorted_data)
        self.img_paths = list(self.img_paths)
        self.labels = list(self.labels)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        # PIL.Image.size → (width, height)
        w, h = image.size
        # crop each dim down to a multiple of 128
        if h % 128 != 0 or w % 128 != 0:
            h = h - (h % 128)
            w = w - (w % 128)
        # center‐crop to (height, width), then to tensor
        crop_and_to_tensor = transforms.Compose([
            transforms.CenterCrop((h, w)),
            transforms.ToTensor()
        ])
        
        return crop_and_to_tensor(image), label

    def __len__(self):
        return len(self.img_paths)
    
    @property
    def num_classes(self):
        return len(self.class_to_idx)


def get_loader(train_dirs, test_dirs, batch_size, num_workers=NUM_DATASET_WORKERS):
    # build datasets using the above GenericImageDataset
    train_ds = GenericImageDataset(train_dirs)
    test_ds  = GenericImageDataset(test_dirs)

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader

def get_classification_loader(train_dir, test_dir=None, batch_size=32, num_workers=NUM_DATASET_WORKERS):
    """
    Create DataLoaders for classification tasks.
    
    Args:
        train_dir: Directory containing training class folders
        test_dir: Directory containing test class folders (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        train_loader, test_loader (or None if test_dir not provided), num_classes, class_to_idx
    """
    # Build datasets using ClassificationImageDataset
    train_ds = ClassificationImageDataset(train_dir)
    
    if test_dir and os.path.exists(test_dir):
        test_ds = ClassificationImageDataset(test_dir)
    else:
        test_ds = None

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

    return train_loader, test_loader, train_ds.num_classes, train_ds.class_to_idx