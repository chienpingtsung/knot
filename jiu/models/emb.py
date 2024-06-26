import einops
import torch
from torch import nn


def one_dim_sincos_pos_emb(emb_dim, pos):
    assert emb_dim % 2 == 0

    k = torch.arange(emb_dim // 2)
    omega = 1 / 10000 ** (2 * k / emb_dim)

    pos = torch.flatten(pos)

    x = torch.einsum('s,c->sc', pos, omega)
    emb_sin = torch.sin(x)
    emb_cos = torch.cos(x)
    emb = torch.cat([emb_sin, emb_cos], dim=1)

    return emb


def two_dim_sincos_pos_emb(emb_dim, h, w, class_token=False):
    assert emb_dim % 2 == 0

    h, w = torch.arange(h), torch.arange(w)
    h, w = torch.meshgrid(h, w, indexing='ij')

    emb_h = one_dim_sincos_pos_emb(emb_dim // 2, h)
    emb_w = one_dim_sincos_pos_emb(emb_dim // 2, w)
    emb = torch.cat([emb_h, emb_w], dim=1)

    if class_token:
        emb = torch.cat([torch.zeros(1, emb_dim), emb], dim=0)

    return emb


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, output_fmt='b (h w) c'):
        super().__init__()

        self.output_fmt = output_fmt

        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)

        if self.output_fmt:
            x = einops.rearrange(x, f'b c h w -> {self.output_fmt}')

        x = self.norm(x)

        return x
