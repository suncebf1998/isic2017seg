from turtle import forward
import einops
from requests import patch
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from functools import partial

# rename na as Non-overlapping CoreCalc Pooling Model(NCP)
class Baisic_Window_Calculation(nn.Module):
    def __init__(self, size, windows_size):
        super().__init__()
        size = to_2tuple(size)
        windows_size = to_2tuple(windows_size)
        match_flag = size[0] % windows_size[0] == 0 and size[1] % windows_size[1] == 0
        assert match_flag, f"image size {size} doesn't match windows size{windows_size}"
        self.size = size
        self.windows_size = windows_size
    
    def _input_transform(self, x):
        B, C, H, W = x.shape
        assert H == self.size[0] and W == self.size[1], \
            f"in fact input(_input_transform) image size ({H}, {W}) doesn't match size {self.size} defined."
        x = einops.rearrange(x, "B C (Hpatch nH) (Wpatch nW) -> (B nH nW) C (Hpatch Wpatch)", Hpatch=self.windows_size[0], Wpatch=self.windows_size[1])
        return x
    
    def _output_transform(self, x, B):
        Bp2, C, L = x.shape
        assert Bp2 == B * (self.size[0] // self.windows_size[0]) * (self.size[1] // self.windows_size[1]), \
            f"in fact input(_output_transform) image size ({Bp2}, {C}, {L}) doesn't match original Batch ({B}) and windowsize {self.windows_size} "
        nH = self.size[0] // self.windows_size[0]
        nW = self.size[1] // self.windows_size[1]
        x = einops.rearrange(
            x, 
            "(B nH nW) C (Hpatch Wpatch) -> B C (Hpatch nH) (Wpatch nW)", 
            Hpatch=self.windows_size[0], Wpatch=self.windows_size[1], nH=nH, nW=nW)
        return x

class Linear_WC(Baisic_Window_Calculation):
    """
    in fact is equal to conv2d
    """
    def __init__(self, size, windows_size, in_channels, out_channels, bias=True, flatten=True):
        super().__init__(size, windows_size)
        self.flatten = flatten
        if self.flatten:
            in_chans = self.windows_size[0] * self.windows_size[1] * in_channels
            out_channels =  self.windows_size[0] *  self.windows_size[1] * out_channels
        # else:
        #     in_chans = (self.size[0] // self.windows_size[0]) * (self.size[1] // self.windows_size[1])
        self.out_dim = out_channels
        self.calc = nn.Linear(in_chans, out_channels, bias=bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x:torch.Tensor = self._input_transform(x)
        if self.flatten:
            x = x.flatten(1, 2)
        else:
            x = x.transpose(-1, -2)
        x = self.calc(x)
        if self.flatten:
            x = x.reshape(x.shape[0], -1, (self.windows_size[0] * self.windows_size[1]))
        else:
            x = x.transpose(-1, -2)
        # print(x.shape)
        x = self._output_transform(x, B)
        return x

class Attention_WC(Baisic_Window_Calculation):
    def __init__(self, size, windows_size, in_channels, out_channels, head_num=8, bias=True, attn_drop=0., proj_drop=0.):
        super().__init__(size, windows_size)
        self.head_num = head_num
        self.qkv = nn.Linear(in_channels, out_channels * 3 * self.head_num, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_channels * self.head_num, out_channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        B, C, H, W = x.shape
        x:torch.Tensor = self._input_transform(x)
        x = x.transpose(-1, -2) # Bp2, L, C
        x = self.qkv(x) # Bp2, L, 3 * newC
        x = x.reshape(x.shape[0], x.shape[1], 3, self.head_num, -1).permute(2, 0, 3, 1, 4) # 3, Bp2, head_num, L, C
        q, k, v = x[0], x[1], x[2]
        B_, _, L, newC = v.shape
        attn = q @ k.transpose(-1, -2) / q.shape[-1]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, L, _ * newC)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1)
        x = self._output_transform(x, B)
        return x

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        # # Number of channels in the training images. For color images this is 3
        # nc = 3

        # # Size of z latent vector (i.e. size of generator input)
        # nz = 100

        # # Size of feature maps in generator
        # ngf = 64
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)




class Down_WC(nn.Module):
    def __init__(self):
        super().__init__()
        




class NCP(nn.Module):
    def __init__(self, in_chans, num_classes, patch_size, embed_dim, dropout=0.2):
        self.embed: nn.Module = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.layer_down: nn.ModuleList = None
        self.head: nn.Module = None
        self.dropout = dropout if dropout is None else nn.Dropout2d(dropout)
        self.seg = nn.Conv2d(embed_dim, num_classes, 1)


    def forward(self, x):
        """
        x: B, C, H, W -- float
        output: B, num_classes, H, W -- float (not p)
        """
        x = self.embed(x)
        outputs = self.downsample(x)
        x = self.upsample(outputs)
        x = self.dembed(x)
        finnal_output = self.cls_seg(self, x)
        return finnal_output

    def downsample(self, x)->list:
        """
        x: B, embed_dim, H, W
        outputs: list of (B, embed_dim, H, W)
        """
        pass
    
    def upsample(self, outputs):
        pass

    def dembed(self, x):
        pass

    def cls_seg(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.seg(x)
    
if __name__ == "__main__":
    layer = Attention_WC((224,224), 4, 3, 16)
    t = torch.randn(16, 3, 224, 224)
    print(layer(t).shape)

    noise = torch.randn(16, 100, 1, 1)
    layer_2 = Generator()
    print(layer_2(noise).shape)

