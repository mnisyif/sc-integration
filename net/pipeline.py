import torch
from build.build_encoder import build_encoder
from net.adapters.llama_adapter import LLaMAAdapter
from net.cloud.llama3 import LLaMAModel

class ModalPipeline:
    """Modular pipeline for different tasks with dynamic resolution support"""
    
    def __init__(self, encoder_name='swin', task='captioning', **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.current_resolution = None
        
        # Build encoder
        print(f"Building encoder: {encoder_name}")
        self.encoder = build_encoder(encoder_name=encoder_name, device=self.device)
        
        # Ensure encoder is on correct device
        self.encoder = self.encoder.to(self.device)
        
        encoder_dim = self.encoder.embed_dims
        print(f"Encoder embed_dim: {encoder_dim}, type: {type(encoder_dim)}")
        
        # Handle multi-stage encoders
        if isinstance(encoder_dim, list):
            encoder_dim = encoder_dim[-1]
            print(f"Using last stage embedding dimension: {encoder_dim}")
        
        # Setup based on task
        if task == 'captioning':
            print("Initializing LLaMA model...")
            self.model = LLaMAModel(kwargs.get('model_name', 'qresearch/llama-3.1-8B-vision-378'))
            llama_dim = self.model.get_embedding_dim()
            print(f"LLaMA embedding dim: {llama_dim}, type: {type(llama_dim)}")
            
            print("Initializing LLaMA adapter...")
            self.adapter = LLaMAAdapter(
                input_dim=encoder_dim,
                llama_dim=llama_dim,
                num_visual_tokens=kwargs.get('num_visual_tokens', 256)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def encode_image(self, image_tensor):
        """Encode image to feature space"""
        # Ensure tensor is on correct device
        image_tensor = image_tensor.to(self.device)
        
        print(f"Input image tensor shape: {image_tensor.shape}")
        print(f"Input image tensor device: {image_tensor.device}")
        print(f"Encoder device: {next(self.encoder.parameters()).device}")
        
        # Encode
        with torch.no_grad():
            features = self.encoder(image_tensor)
        
        print(f"Encoded features shape: {features.shape}")
        return features
    
    def process(self, image_tensor, **kwargs):
        """Full pipeline: encode -> adapt -> model"""
        # Encode image
        features = self.encode_image(image_tensor)
        
        # Adapt features
        adapted_features = self.adapter(features)
        
        # Process with model
        if self.task == 'captioning':
            prompt = kwargs.get('prompt', 'Caption the image')
            max_tokens = kwargs.get('max_tokens', 128)
            return self.model.generate_from_features(adapted_features, prompt, max_tokens)
    
    def encode_and_adapt(self, image_tensor):
        """Encode and adapt for external use"""
        features = self.encode_image(image_tensor)
        adapted_features = self.adapter(features)
        return adapted_features