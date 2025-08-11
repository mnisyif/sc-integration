# import torch
# import torch.nn as nn
# from timm.layers.weight_init import trunc_normal_

# class SwinClassifier(nn.Module):
#     def __init__(self, encoder_dim, num_classes, drop_rate=0.0):
#         super().__init__()
#         self.encoder_dim = encoder_dim
#         self.num_classes = num_classes

#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#         if drop_rate > 0.0:
#             self.dropout = nn.Dropout(drop_rate)
#         else:
#             self.dropout = nn.Dropout(0.0)

#         # classification head
#         self.head = nn.Linear(encoder_dim, num_classes)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         B, L, C = x.shape

#         x = x.transpose(1, 2)  # (B, C, L)
#         x = self.global_pool(x) # (B, C, 1)
#         x = x.squeeze(-1) # (B, C)

#         x = self.dropout(x)
#         x = self.head(x)

#         return x

#     def flopt(self, input_resolution):
#         H, W = input_resolution
#         flops = H * W * self.encoder_dim * self.num_classes
#         return flops

import torch.nn.functional as F
import torch.nn as nn
import torch

from timm.layers.weight_init import trunc_normal_

class GenericClassifier(nn.Module):
    """
    Handles both (B, N, C) and (B, C, H', W') encoder outputs.
    Adds a projection layer to reduce high-dimensional features.
    """
    def __init__(self, in_features: int, num_classes: int, hidden: int = 512, agg: str = "mean", 
                 projection_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        assert agg in {"mean", "cls", "max", "gap"}
        self.agg = agg
        
        # Add projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classifier with reduced input dimension
        self.fc = nn.Sequential(
            nn.Linear(projection_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # --- unify shapes to (B, C) ----------------------------------------
        if x.dim() == 4:  # CNN: (B, C, H', W')
            if self.agg == "gap" or self.agg == "mean":
                x = x.mean(dim=(2, 3))  # Global Average Pool
            elif self.agg == "max":
                x = x.amax(dim=(2, 3))
        elif x.dim() == 3:  # Transformer: (B, N, C)
            if self.agg == "cls":
                x = x[:, 0]  # assume first token is [CLS]
            elif self.agg == "max":
                x, _ = x.max(dim=1)
            else:  # "mean"
                x = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported shape {x.shape}")
        
        # Apply projection to reduce dimensionality
        x = self.projection(x)
        
        # Final classification
        return self.fc(x)  # (B, num_classes)
    # """
    # Handles both (B, N, C) and (B, C, H', W') encoder outputs.
    # Choose how to collapse the spatial / token dimension via `agg`.
    # """
    # def __init__(self, in_features: int, num_classes: int,
    #              hidden: int = 512, agg: str = "mean"):
    #     super().__init__()
    #     assert agg in {"mean", "cls", "max", "gap"}
    #     self.agg = agg
    #     self.fc = nn.Sequential(
    #         nn.Linear(in_features, hidden),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(hidden, num_classes)
    #     )

    # def forward(self, x):
    #     # --- unify shapes to (B, C) ----------------------------------------
    #     if x.dim() == 4:                      # CNN: (B, C, H', W')
    #         if self.agg == "gap" or self.agg == "mean":
    #             x = x.mean(dim=(2, 3))        # Global Average Pool
    #         elif self.agg == "max":
    #             x = x.amax(dim=(2, 3))
    #     elif x.dim() == 3:                    # Transformer: (B, N, C)
    #         if self.agg == "cls":
    #             x = x[:, 0]                  # assume first token is [CLS]
    #         elif self.agg == "max":
    #             x, _ = x.max(dim=1)
    #         else:  # "mean"
    #             x = x.mean(dim=1)
    #     else:
    #         raise ValueError(f"Unsupported shape {x.shape}")
    #     # -------------------------------------------------------------------
    #     return self.fc(x)                    # (B, num_classes)

class AttentionPoolingClassifier(nn.Module):
    """
    Alternative approach using attention-based pooling
    """
    def __init__(self, in_features: int, num_classes: int,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        
        # Attention mechanism for pooling
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        elif x.dim() == 3:  # (B, N, C)
            pass
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Attention-weighted pooling
        attention_weights = self.attention(x)  # (B, N, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (B, N, 1)
        
        # Weighted sum of features
        x = torch.sum(x * attention_weights, dim=1)  # (B, C)
        
        # Project and classify
        x = self.feature_proj(x)
        return self.classifier(x)

class SwinClassifier(nn.Module):
    """
    Swin Classifier that takes latent representations from a pretrained encoder
    and performs classification.
    """
    def __init__(self, encoder_dim, num_classes, dropout=0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        
        # Global average pooling to reduce sequence dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, latent_features):
        """
        Args:
            latent_features: (B, L, C) - latent features from encoder
        Returns:
            logits: (B, num_classes) - classification logits
        """
        # latent_features shape: (B, L, C) where L is sequence length
        # Global average pooling: (B, L, C) -> (B, C, L) -> (B, C, 1) -> (B, C)
        x = latent_features.transpose(1, 2)  # (B, C, L)
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits