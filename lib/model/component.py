import torch
from einops import rearrange, repeat
from torch import nn
from torchvision import ops


def conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
                         ops.FrozenBatchNorm2d(out_ch) if freeze_bn else nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))


class CornerPredictor(nn.Module):
    def __init__(self, in_ch, patch_num, patch_size, freeze_bn=False):
        super().__init__()

        self.conv_1 = conv(in_ch, in_ch // 2, freeze_bn=freeze_bn)
        self.conv_2 = conv(in_ch // 2, in_ch // 4, freeze_bn=freeze_bn)
        self.conv_3 = conv(in_ch // 4, in_ch // 8, freeze_bn=freeze_bn)
        self.conv_4 = conv(in_ch // 8, in_ch // 16, freeze_bn=freeze_bn)
        self.conv_5 = nn.Conv2d(in_ch // 16, 2, kernel_size=1)

        index = torch.arange(patch_num) * patch_size
        self.coord_x = nn.Parameter(repeat(index, 'x -> (repeat x)', repeat=patch_num), requires_grad=False)
        self.coord_y = nn.Parameter(repeat(index, 'y -> (y repeat)', repeat=patch_num), requires_grad=False)

    def softmax(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = torch.softmax(x, dim=1)
        exp_x = torch.sum(self.coord_x * x, dim=1)
        exp_y = torch.sum(self.coord_y * x, dim=1)
        return exp_x, exp_y

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        tl, br = torch.split(x, 1, dim=1)
        tl_x, tl_y = self.softmax(tl)
        br_x, br_y = self.softmax(br)

        return torch.stack((tl_x, tl_y, br_x, br_y), dim=1), x
