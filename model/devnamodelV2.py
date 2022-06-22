import einops
from requests import patch
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from functools import partial

# rename na as Non-overlapping CoreCalc Pooling Model(NCP)
class Baisic_Window_Calculation(nn.Module):
    def __init__(self, size, windows_size):
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
        x = einops.rearrange(x, "B C (Hpatch nH) (Wpatch nW) -> (B Hpatch Wpatch) C (nH nW)", Hpatch=self.windows_size[0], Wpatch=self.windows_size[1])
        return x
    
    def _output_transform(self, x, B):
        Bp2, C, L = x.shape
        assert Bp2 == B * self.windows_size[0] * self.windows_size[1], \
            f"in fact input(_output_transform) image size ({Bp2}, {C}, {L}) doesn't match original Batch ({B}) and windowsize {self.windows_size} "
        nH = self.size[0] // self.windows_size[0]
        nW = self.size[1] // self.windows_size[1]
        x = einops.rearrange(
            x, 
            "(B Hpatch Wpatch) C (nH nW) -> B C (Hpatch nH) (Wpatch nW)", 
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
            in_chans = (self.size[0] // self.windows_size[0]) * (self.size[1] // self.windows_size[1]) * in_channels
        else:
            in_chans = (self.size[0] // self.windows_size[0]) * (self.size[1] // self.windows_size[1])
        self.calc = nn.Linear(in_chans, out_channels, bias=bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x:torch.Tensor = self._input_transform(x)
        x = x.flatten(1, 2)
        x = self.calc(x)
        x = x.reshape(x.shape[0], -1, H * W)
        x = self._output_transform(x, B)
        return x

class Attention_WC(Baisic_Window_Calculation):
    pass




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
    