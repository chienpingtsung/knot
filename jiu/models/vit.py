from functools import partial

import einops
import timm
import torch
from torch import nn

from jiu.models.emb import PatchEmbed, two_dim_sincos_pos_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.qkv(x)
        x = einops.rearrange(x, 'b s (qkv heads c) -> qkv b heads s c', qkv=3, heads=self.num_heads)
        q, k, v = x.unbind()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = einops.rearrange(x, 'b heads s c -> b s (heads c)')
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0, drop_path=0, mlp_ratio=4,
                 norm_layer=nn.LayerNorm, mlp_layer=timm.layers.Mlp, act_layer=nn.GELU):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.drop_path1 = timm.layers.DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer, drop=proj_drop)
        self.drop_path2 = timm.layers.DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, qkv_bias=True, class_token=True, proj_drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer=None, act_layer=nn.GELU, block_fn=Block, mlp_layer=timm.layers.Mlp):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, 'b (h w) c')
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        h = w = img_size // patch_size
        self.pos_embed = nn.Parameter(two_dim_sincos_pos_emb(embed_dim, h, w, class_token))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[block_fn(dim=embed_dim,
                                               num_heads=num_heads,
                                               qkv_bias=qkv_bias,
                                               attn_drop=attn_drop_rate,
                                               proj_drop=proj_drop_rate,
                                               drop_path=dpr[i],
                                               mlp_ratio=mlp_ratio,
                                               norm_layer=norm_layer,
                                               mlp_layer=mlp_layer,
                                               act_layer=act_layer) for i in range(depth)])

        self.norm = norm_layer(embed_dim) if num_classes else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, x):
        b, *_ = x.shape

        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)

        if self.num_classes:
            x = self.norm(x)
            x = self.head(x[:, 0, ...])

        return x
