import torch, datetime
import torch.nn as nn

from timm.layers.weight_init import trunc_normal_

from net.architectures.swin import SwinTransformerBlock, PatchReverseMerging

class BasicLayer_Decoder(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i%2==0) else window_size // 2), mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample:
            x = self.upsample(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = sum(blk.flops() for blk in self.blocks) # type: ignore
        if self.upsample:
            flops += self.upsample.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for blk in self.blocks:
            blk.input_resolution = (H, W) # type: ignore
            blk.update_mask() # type: ignore
        if self.upsample:
            self.upsample.input_resolution = (H, W)

class SwinDecoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, window_size=4, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.num_layers = len(depths)
        self.H = img_size[0]
        self.W = img_size[1]
        self.patch_norm = patch_norm
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.patches_res = (self.H // (2**self.num_layers), self.W // (2**self.num_layers))
        self.hidden_dim = int(embed_dims[0] * 1.5)
        self.layer_num = 7

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer_Decoder(dim=int(embed_dims[i]), out_dim=int(embed_dims[i+1] if (i < self.num_layers - 1) else 3),
                                       input_resolution=(self.patches_res[0] * (2**i), self.patches_res[1] * (2**i)),
                                       depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio,
                                       qkv_bias=True, qk_scale=None, norm_layer=norm_layer, upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        print()
        self.apply(self._init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_resolution(self, H, W):
        self.input_resolution = (H, W) # type: ignore
        self.H = H * (2 ** self.num_layers)
        self.W = W * (2 ** self.num_layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer), W * (2 ** i_layer)) # type: ignore

    def flops(self):
        return sum(layer.flops() for layer in self.layers) # type: ignore

def create_decoder(**kwargs):
    model = SwinDecoder(**kwargs)
    return model