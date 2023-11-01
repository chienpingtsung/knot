from functools import partial

import torch
from einops import rearrange
from timm import layers, models
from torch import nn

from lib.model.component import PatchEmbed, two_dim_sincos_pos_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()

        assert dim % num_heads == 0, 'Attention dim should be divisible by num_heads.'

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.qkv(x)
        x = rearrange(x, 'b s (qkv head c) -> qkv b head s c', qkv=3, head=self.num_heads)
        q, k, v = torch.unbind(x)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = rearrange(x, 'b head s c -> b s (head c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0, path_drop=0, mlp_ratio=4,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, attn_layer=Attention):
        super().__init__()

        self.norm_1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.pdrp_1 = layers.DropPath(path_drop) if path_drop > 0 else nn.Identity()

        self.norm_2 = norm_layer(dim)
        self.mlp = layers.Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.pdrp_2 = layers.DropPath(path_drop) if path_drop > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.pdrp_1(self.attn(self.norm_1(x)))
        x = x + self.pdrp_2(self.mlp(self.norm_2(x)))
        return x


class VisionTransformer(models.VisionTransformer):
    def __init__(self,
                 img_size_t=128, img_size_s=288,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0,
                 proj_drop_rate=0,
                 attn_drop_rate=0,
                 drop_path_rate=0,
                 weight_init='',
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 block_fn=Encoder):
        super().__init__(patch_size=patch_size,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop_rate=drop_rate,
                         proj_drop_rate=proj_drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer,
                         act_layer=act_layer)
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(patch_size, in_chans, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoders = nn.Sequential(*[block_fn(embed_dim,
                                                 num_heads,
                                                 qkv_bias,
                                                 attn_drop_rate,
                                                 proj_drop_rate,
                                                 dpr[i],
                                                 mlp_ratio,
                                                 norm_layer,
                                                 act_layer) for i in range(depth)])

        self.patch_num_t = img_size_t // patch_size
        self.patch_num_s = img_size_s // patch_size
        self.seq_len_t = self.patch_num_t ** 2
        self.seq_len_s = self.patch_num_s ** 2

        self.pos_embed_t = nn.Parameter(two_dim_sincos_pos_emb(embed_dim, self.patch_num_t, self.patch_num_t),
                                        requires_grad=False)
        self.pos_embed_s = nn.Parameter(two_dim_sincos_pos_emb(embed_dim, self.patch_num_s, self.patch_num_s),
                                        requires_grad=False)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward(self, x_t, x_s):
        x_t = self.patch_embed(x_t) + self.pos_embed_t
        x_s = self.patch_embed(x_s) + self.pos_embed_s
        x = torch.cat([x_t, x_s], dim=1)
        x = self.pos_drop(x)

        x = self.encoders(x)

        x_t, x_s = torch.split(x, [self.seq_len_t, self.seq_len_s], dim=1)
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=self.patch_num_t, w=self.patch_num_t)
        x_s = rearrange(x_s, 'b (h w) c -> b c h w', h=self.patch_num_s, w=self.patch_num_s)

        return x_t, x_s
