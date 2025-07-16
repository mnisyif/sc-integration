import torch
import torch.nn as nn

from timm.layers.weight_init import trunc_normal_

from net.architectures.swin import SwinTransformerBlock, PatchMerging, PatchEmbed

class BasicLayer_Encoder(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_dim, input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2), 
                num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def update_resolution(self, H, W):
        for blk in self.blocks:
            blk.input_resolution = (H,W) # type: ignore
            blk.update_mask() # type: ignore
        if self.downsample is not None:
            self.downsample.input_resolution = (H*2, W*2)

class SwinEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dims, depths, num_heads, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.hidden_dim = int(self.embed_dims[-1] * 1.5) #embed_dims[-1] * 1.5
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        self.layer_num = len(depths) * 2 - 1         
        self.patches_res = img_size

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_Encoder(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3, out_dim=int(embed_dims[i_layer]),
                input_resolution=(self.patches_res[0] // (2 ** i_layer), self.patches_res[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != 0 else None
            )
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        print()
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # x = self.head_list(x) # type: ignore
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_resolution(self, H:int, W:int):
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)), W // (2 ** (i_layer + 1))) # type: ignore
    
    def flops(self):
        flops = self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops() # type: ignore
        return flops

def create_encoder(**kwargs):
    model = SwinEncoder(**kwargs)
    return model
