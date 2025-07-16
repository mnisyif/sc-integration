import torch
import torch.nn as nn
from net.interfaces.base_adaptor import BaseAdapter

class LLaMAAdapter(BaseAdapter):
    """Adapter for LLaMA vision models"""
    
    def __init__(self, input_dim, llama_dim, num_visual_tokens=256):
        # Handle different input_dim types
        if isinstance(input_dim, list):
            input_dim = input_dim[-1]  # Use the last dimension
        
        # Debug: Print dimensions to check for issues
        print(f"LLaMAAdapter init - input_dim: {input_dim}, llama_dim: {llama_dim}, num_visual_tokens: {num_visual_tokens}")
        print(f"Types - input_dim: {type(input_dim)}, llama_dim: {type(llama_dim)}")
        
        super().__init__(input_dim, llama_dim)
        self.num_visual_tokens = num_visual_tokens
        
        # Ensure dimensions are integers
        input_dim = int(input_dim)
        llama_dim = int(llama_dim)
        
        # Feature projection layers
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, llama_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llama_dim * 2, llama_dim),
            nn.LayerNorm(llama_dim)
        )
        
        # Learnable visual tokens for compression
        self.visual_tokens = nn.Parameter(torch.randn(1, num_visual_tokens, llama_dim))
        
        # Cross-attention to compress features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llama_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, features):
        """
        Transform Swin features (B, N, C) to LLaMA format
        """
        B, N, C = features.shape
        print(f"LLaMAAdapter forward - features shape: {features.shape}")
        
        # Project features to LLaMA embedding space
        projected_features = self.feature_projector(features)  # (B, N, llama_dim)
        print(f"Projected features shape: {projected_features.shape}")
        
        # Use cross-attention to compress to fixed number of visual tokens
        visual_tokens = self.visual_tokens.expand(B, -1, -1)  # (B, num_visual_tokens, llama_dim)
        
        adapted_features, attention_weights = self.cross_attention(
            visual_tokens, projected_features, projected_features
        )
        
        print(f"Adapted features shape: {adapted_features.shape}")
        return adapted_features  # (B, num_visual_tokens, llama_dim)
    
    def get_output_shape(self, input_shape):
        B, N, C = input_shape
        return (B, self.num_visual_tokens, self.output_dim)