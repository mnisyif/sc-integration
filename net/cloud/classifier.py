import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_

class SwinClassifier(nn.Module):
    def __init__(self, encoder_dim, num_classes, drop_rate=0.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if drop_rate > 0.0:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = nn.Dropout(0.0)

        # classification head
        self.head = nn.Linear(encoder_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape

        x = x.transpose(1, 2)  # (B, C, L)
        x = self.global_pool(x) # (B, C, 1)
        x = x.squeeze(-1) # (B, C)

        x = self.dropout(x)
        x = self.head(x)

        return x

    def flopt(self, input_resolution):
        H, W = input_resolution
        flops = H * W * self.encoder_dim * self.num_classes
        return flops