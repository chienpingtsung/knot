import functools
import math

import einops
import timm
import torch
from torch import nn

from jiu.models.emb import PatchEmbed


def window_partition(x, win_h, win_w):
    x = einops.rearrange(x, 'b (nwh wh) (nww ww) c -> (b nwh nww) wh ww c', wh=win_h, ww=win_w)
    return x


def window_reverse(x, win_h, win_w, h, w):
    x = einops.rearrange(x, '(b nwh nww) wh ww c -> b (nwh wh) (nww ww) c', nwh=h // win_h, nww=w // win_w)
    return x


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w), indexing='ij'))
    coords = torch.flatten(coords, 1)
    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0, proj_drop=0):
        super().__init__()

        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads

        self.num_heads = num_heads

        self.scale = head_dim ** -0.5

        win_h = win_w = window_size
        self.window_area = win_h * win_w
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        self.register_buffer('relative_position_index', get_relative_position_index(win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.window_area, self.window_area, -1).permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask=None):
        b, s, c = x.shape

        x = self.qkv(x)
        x = einops.rearrange(x, 'b s (qkv heads c) -> qkv b heads s c', qkv=3, heads=self.num_heads)
        q, k, v = x.unbind()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win, *_ = mask.shape
            attn = attn.view(-1, num_win, self.num_heads, s, s) + mask.unsqueeze(1).unsqueeze(0).to(attn.device)
            attn = attn.view(-1, self.num_heads, s, s)
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = einops.rearrange(x, 'b heads s c -> b s (heads c)')
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=None, window_size=7, shift_size=0, drop_path=0,
                 qkv_bias=True, attn_drop=0, proj_drop=0, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.window_size = (window_size, window_size)
        self.shift_size = (shift_size, shift_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim=dim,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop)
        self.drop_path1 = timm.layers.DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = timm.layers.Mlp(dim, int(mlp_ratio * dim), act_layer=act_layer, drop=proj_drop)
        self.drop_path2 = timm.layers.DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _attn(self, x):
        b, h, w, c = x.shape
        shift_size = (self.shift_size[0] if h > self.window_size[0] else 0,
                      self.shift_size[1] if w > self.window_size[1] else 0)

        has_shift = any(shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        shifted_x = nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))

        x_windows = window_partition(shifted_x, *self.window_size)
        x_windows = einops.rearrange(x_windows, 'b h w c -> b (h w) c')

        if has_shift:
            attn_mask = self.generate_attn_mask(h + pad_h, w + pad_w, shift_size)
        else:
            attn_mask = None
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = einops.rearrange(attn_windows, 'b (h w) c -> b h w c',
                                        h=self.window_size[0], w=self.window_size[1])
        shifted_x = window_reverse(attn_windows, *self.window_size, h + pad_h, w + pad_w)
        shifted_x = shifted_x[:, :h, :w, :].contiguous()

        if has_shift:
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2))
        else:
            x = shifted_x

        return x

    @functools.lru_cache()
    def generate_attn_mask(self, h, w, shift_size):
        h = math.ceil(h / self.window_size[0]) * self.window_size[0]
        w = math.ceil(w / self.window_size[1]) * self.window_size[1]
        img_mask = torch.zeros(1, h, w, 1)
        cnt = 0
        for hs in (slice(0, -self.window_size[0]),
                   slice(-self.window_size[0], -shift_size[0]),
                   slice(-shift_size[0], None)):
            for ws in (slice(0, -self.window_size[1]),
                       slice(-self.window_size[1], -shift_size[1]),
                       slice(-shift_size[1], None)):
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, *self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        b, h, w, c = x.shape
        x = x + self.drop_path1(self._attn(self.norm1(x)))
        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()

        out_dim = out_dim or 2 * dim

        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        b, h, w, c = x.shape
        assert h % 2 == 0
        assert w % 2 == 0

        x = einops.rearrange(x, 'b (nh h) (nw w) c -> b nh nw (w h c)', nh=h // 2, nw=w // 2)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage(nn.Module):
    def __init__(self, dim, out_dim, depth, downsample=True, num_heads=4, head_dim=None, window_size=7, drop_path=0,
                 qkv_bias=True, attn_drop=0, proj_drop=0, mlp_ratio=4, norm_layer=nn.LayerNorm):
        super().__init__()

        shift_size = window_size // 2

        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(*[SwinTransformerBlock(dim=out_dim,
                                                           num_heads=num_heads,
                                                           head_dim=head_dim,
                                                           window_size=window_size,
                                                           shift_size=0 if (i % 2 == 0) else shift_size,
                                                           drop_path=drop_path[i] if isinstance(
                                                               drop_path, (tuple, list)
                                                           ) else drop_path,
                                                           qkv_bias=qkv_bias,
                                                           attn_drop=attn_drop,
                                                           proj_drop=proj_drop,
                                                           mlp_ratio=mlp_ratio,
                                                           norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), head_dim=None, window_size=7, mlp_ratio=4, qkv_bias=True, drop_rate=0,
                 proj_drop_rate=0, attn_drop_rate=0, drop_path_rate=0.1, embed_layer=PatchEmbed,
                 norm_layer=nn.LayerNorm, weight_init='', **kwargs):
        super().__init__()

        self.num_layers = len(depths)

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.patch_embed = embed_layer(patch_size=patch_size, in_channels=in_chans, emb_dim=embed_dim[0], flatten=False)
        self.patch_norm = norm_layer(embed_dim[0])

        if not isinstance(head_dim, (tuple, list)):
            head_dim = [head_dim] * self.num_layers
        if not isinstance(window_size, (tuple, list)):
            window_size = [window_size] * self.num_layers
        if not isinstance(mlp_ratio, (tuple, list)):
            mlp_ratio = [mlp_ratio] * self.num_layers
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers.append(SwinTransformerStage(dim=in_dim,
                                               out_dim=out_dim,
                                               depth=depths[i],
                                               downsample=i > 0,
                                               num_heads=num_heads[i],
                                               head_dim=head_dim[i],
                                               window_size=window_size[i],
                                               drop_path=dpr[i],
                                               qkv_bias=qkv_bias,
                                               attn_drop=attn_drop_rate,
                                               proj_drop=proj_drop_rate,
                                               mlp_ratio=mlp_ratio[i],
                                               norm_layer=norm_layer))
            in_dim = out_dim
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(embed_dim[-1])
        self.head = timm.layers.ClassifierHead(embed_dim[-1],
                                               num_classes,
                                               pool_type=global_pool,
                                               drop_rate=drop_rate,
                                               input_fmt='NHWC')

    def forward(self, x):
        x = self.patch_embed(x)
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.patch_norm(x)

        x = self.layers(x)

        x = self.norm(x)
        x = self.head(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd