""" Parts of the U-Net model """
# cite: https://github.com/milesial/Pytorch-UNet 

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode

class WeightConv2d(nn.Conv2d):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input):
        d1, d2, h, w = self.weight.shape
        weight = self.weight.flatten(-2, -1)
        weight = self.softmax(weight)
        weight = weight.reshape(d1, d2, h, w)
        return self._conv_forward(input, weight, self.bias)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            WeightConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            WeightConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_Up(nn.Module):
    """Downscaling and Upscaling with maxpool and repeat"""

    def __init__(self, ratio=2, size=(224, 224)):
        super().__init__()
        self.size = size
        self.down = nn.AvgPool2d(ratio)
        self.up = nn.UpsamplingNearest2d(scale_factor=ratio)
    
    def forward(self, x):
        assert x.shape[2] == self.size[0] and x.shape[3] == self.size[1], \
             f"input x's shape ({x.shape[2]}, {x.shape[3]}) must be the same with size {self.size}"
        return self.up(self.down(x))



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
