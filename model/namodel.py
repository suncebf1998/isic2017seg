# --------------------------------------------------------
# NA model
# Copyright (c) 2022 CPU
# Licensed under The MIT License [see LICENSE for details]
# Written by Chenhu Sun
# dev.py: develop the NA model
# --------------------------------------------------------

# from modulefinder import Module
import einops
from requests import patch
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from functools import partial
__all__ = ("NonAver",)


"""
NA model
Non-overlapping Windows  + Average Pool Model
Non-overlapping Windows:
\begin{aligned}
    Windows(Input) &= [w1, w2, w3, ...]\\
    w_{i} \cup w_{j} &= \varnothing (i \ne j)\\
    \cup_{i=1}^{n}(w_{i}) &= Input\\
    \cup_{i=1}^{n}(Core(w_{i})) &= Output
\end{aligned} \tag{1}
Average Pool:
    AveragePool2D(Input) = pooled
    repeat(pooled) = Output
Cat:
    ->cat(Outputs)
    ->conv2d(size=(1, 1))
"""


class AverProj(nn.Module):
    """
    with out trainable params layer
    """
    def __init__(self, ratio:int = 2, model="image", *, size=None):
        assert model in ("patch", "image"), f"model must be one of 'patch' and 'image'"
        super().__init__()
        ratio = int(ratio)
        if ratio == -1:
            assert size is not None, f"ratio is {ratio}, meaning using the global average pool, while the size of input is not specified."
            self.downsample = nn.AvgPool2d(size)
            ratio = size
        elif ratio < 1:
            raise AttributeError("ratio must be larger or equal than 1, while ratio = {} now!".format(ratio))
        elif ratio > 1:
            if model == "patch":
                self.downsample = nn.AvgPool1d(ratio)
            elif model == "image":
                self.downsample = nn.AvgPool2d(ratio)
        else:
            self.downsample = nn.Identity()
        
        self.model = model == "patch"
        if not self.model:
            self.ratio = to_2tuple(ratio)
        else:
            self.ratio = ratio

        
    def forward(self, x):
        # x: B, dim, h, w
        # out: sample as x
        if self.ratio == 1:
            return self.downsample(x)
        else:
            x = self.downsample(x) # B, dim, h/ratio, w/ratio
            if self.model:
                x = einops.repeat(x, "b c s -> (s rs)", rs=self.ratio)
            else:
                x = einops.repeat(x, "b c h w -> b c (h rh) (w rw)", rh=self.ratio[0], rw=self.ratio[1])
            # x = einops.repeat(x, "b c h w -> b c h (w repeat)", repeat=self.ratio[1])
            return x


class MultiAver(nn.Module):
    def __init__(self, ratios=[2, 4], unit=False, proj=None, proj_out=None, size=None, model="image"):
        """
        model: "patch" or "image"
        """
        super().__init__()
        self.downs = nn.ModuleDict()
        for ratio in ratios:
            layer_name = "Down_X" + str(ratio)
            layer = AverProj(ratio=ratio, size=size, model=model)
            self.downs[layer_name] = layer
        self.unit = unit
        self.proj = nn.Identity() if proj is None else proj
        self.proj_out = nn.Identity() if proj_out is None else proj_out

    def forward(self, x):
        x = self.proj(x)
        features = [x, ]
        ##
        _count = 0
        ##
        for averproj in self.downs.values():
            f = averproj(x)
            f = self.proj_out(f)
            features.append(f)
        if self.unit:
            features = torch.cat(features, 1)
        return features


class Mlp(nn.Module):
    # FADFD work on the last dim
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., output=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.output = output
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.output:
            self.sigmod = nn.Sigmoid()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.output:
            x = self.sigmod(x)
        else:
            x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, proj=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size

        if proj is None:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = proj(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: B C(in_chans) H W (to_2tuple(img_size))
        # output: B embed_dim H W
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) ##  .flatten(2).transpose(1, 2)  # B Ph*Pw C 
        return x

class MlpEx(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0., use_channel_last=False, output=False):
        super().__init__()
        self.use_channel_last = use_channel_last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features,output=output)
        self.norm = nn.LayerNorm(out_features) # 


    def forward(self, x: torch.Tensor):
        # x: B C H W or B H W C
        # out: B out_features H W or B H W out_features
        if not self.use_channel_last:
            x = x.permute(0, 2, 3, 1) # B H W C
        ##
        # x = x + self.drop_path(self.mlp(x))
        x = self.drop_path(self.mlp(x))
        ##
        x = self.norm(x)
        if not self.use_channel_last:
            x = x.permute(0, 3, 1, 2) # B
        return x

class NonOverlapping(nn.Module):
    def __init__(self, core, patchsize, use_skip=True):
        super().__init__()
        self.core = core
        self.patchsize = patchsize
        self.use_skip = use_skip

    def forward(self, x):
        # x: B C H W
        # output: B C H W (same as x) or (B nH nW) (pH pW) C'
        B, C, H, W = x.shape
        assert (H / self.patchsize[0] == H // self.patchsize[0]) and (W / self.patchsize[1] == W // self.patchsize[1]), \
            f"Input image's height or weight ({H}*{W}) isn't patchsize({self.patchsize})'s multiple."
        x = einops.rearrange(x, "B C (nH pH) (nW pW) -> (B nH nW) (pH pW) C", pH=self.patchsize[0], pW=self.patchsize[1])
        if not self.use_skip:
            x = self.core(x) 
            return x
        x = x + self.core(x)
        x = einops.rearrange(
            x, "(B nH nW) (pH pW) C-> B C (nH pH) (nW pW)", 
            B=B, nH=H // self.patchsize[0], nW=W // self.patchsize[1], pH=self.patchsize[0], pW=self.patchsize[1])
        return x

class MatCore(nn.Module):
    def __init__(self, matsize, out_features=None, activation=nn.GELU) -> None:
        super().__init__()
        # N, C = matsize
        out_features = matsize[1] if out_features is None else out_features
        self.mproj = nn.Linear(in_features=matsize[1], out_features=matsize[1])
        self.softmax = nn.Softmax(dim=-1)
        self.nproj = nn.Linear(in_features=matsize[0], out_features=out_features)
        self.act = nn.Identity() if activation is None else activation()


    def forward(self, x):
        # x: B N C
        # output: B N C'
        """
        Latex:
            \begin{aligned}
                &X = X \cdot M \cdot X^{T} \cdot N \\
                &X = \text{activation}(X)\\
                &X:B, (N, C)\\
                &M: (C, C)\\
                &X^{T}: B, (C, N)\\
                &N: (N, C_{\text{out}})
            \end{aligned}
        """
        ##
        x = self.softmax(self.mproj(x)) @ x.transpose(-2, -1) 
        ##
        # ori = x
        # x = self.mproj(x)
        # x = x @ ori.transpose(-2, -1) 
        ##
        x = self.nproj(x)
        x = self.act(x)
        return x

class AttentionCore(MatCore):

    def __init__(self, matsize):
        super().__init__(matsize, matsize[0], activation=partial(nn.Softmax, dim=-1))

    def forward(self, x):
        x = super().forward(x) @ x
        return x


class NonAverLoop(nn.Module):
    """
    Loop:
        2.1 NonOverlapping # B C H W -> B C H W
        2.2 MultiAver # B C H W -> B (len(ratios) + 1)*C H W
        2.3 MlpEx # B (len(ratios) + 1)*C H W -> B C H W
    """

    def __init__(self, embed_dim, patch_size=4, ratios=[2, 4, 8], out_features=None):
        super().__init__()
        pH, pW = patch_size
        C = embed_dim
        out_features = out_features or C
        self.core = MatCore(matsize=(pH * pW, C))
        self.nonover = NonOverlapping(self.core, patch_size)
        self.averpool = MultiAver(ratios, True)
        self.mlps = MlpEx((len(ratios) + 1) * C, out_features=C)
        

    def forward(self, x):
        # x: B H W C
        # output: B H W C'
        x = self.nonover(x)
        x = self.averpool(x)
        x = self.mlps(x)
        return x

class NonAverExpand(nn.Module):
    """
    Loop:
        2.1 NonOverlapping # B C H W -> B C H W
        2.2 MultiAver # B C H W -> B (len(ratios) + 1)*C H W
        2.3 MlpEx # B (len(ratios) + 1)*C H W -> B C H W
    """

    def __init__(self, size, embed_dim, patch_size, ratios=[2, 4, 8, -1], out_features=None, expand=None):
        super().__init__()
        pH, pW = patch_size
        C = embed_dim
        out_features = out_features or C
        self.expand = expand or patch_size
        self.size = size
        self.path_size =patch_size
        self.nH = size[0] // patch_size[0]
        self.nW = size[1] // patch_size[1]
        self.pH = patch_size[0]
        self.pW = patch_size[1]
        self.outsize = size[0] * self.expand[0] , size[1] * self.expand[1]
        core = MatCore(matsize=(pH * pW, C), out_features=self.expand[0] * self.expand[1] * C)
        self.nonover = NonOverlapping(core, patch_size, use_skip=False)
        self.averpool = MultiAver(ratios, True, size=self.outsize)
        self.mlps = MlpEx((len(ratios) + 1) * C, out_features=out_features, output=True)
        

    def forward(self, x):
        # x: B C H W 
        # output: B C' H W 
        B, C, H, W = x.shape
        assert H * W == self.size[0] * self.size[1], f"Input ({H}, {W}) doesn't match size ({self.size})"
        x = self.nonover(x) # (B nH nW) (pH pW) (C * expand[0] * expand[1])
        x = einops.rearrange(
            x, "(B nH nW) (pH pW) (C eH eW) -> B C (nH pH eH) (nW pW eW)", 
            B=B, nH=self.nH, nW=self.nW, pH=self.pH, pW=self.pW, eH=self.expand[0], eW=self.expand[1])
        x = self.averpool(x)
        x = self.mlps(x)
        return x

class NonAver(nn.Module):
    """
    1. PatchEmbed
    2. Loop:
        2.1 NonOverlapping
        2.2 MultiAver
        2.3 MlpEx
    3. Mission
    """

    def __init__(self, img_size=224, in_chans=3, embed_dim=96, patch_size=4, num_classes=1, ratios=[2, 4, 8], repeat=3, *args, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.nas = nn.ModuleList()
        for _ in range(repeat):
            self.nas.append(NonAverLoop(embed_dim, patch_size, ratios))
        self.exseg = NonAverExpand(size, embed_dim, patch_size, out_features=num_classes, expand=[4, 4])
        
    # def forward(self, x):
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embed(x)
        x = self.nasward(x) # B C H//pH W//pW
        x = self.exseg(x)
        return x

    def nasward(self, x):
        for na in self.nas:
            x = na(x)
        return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 可以让程序报错时显示的更加human
    print("---test na model---")
    # model = NonAverLoop((4,4),96).cuda() # NonAver().cuda()
    # model = nn.ModuleList()
    # for i in range(3):
    #     layer =nn.Identity() # NonAverLoop((4,4),96)
    #     model.append(layer)
    # model = model.cuda()
    # # print(model)
    # test = torch.randn(16, 96, 56, 56).cuda()

    model = NonAver().cuda()
    test = torch.ones(16, 3, 224, 224, dtype=torch.float32).cuda()

    print(model(test).max())





    

        

