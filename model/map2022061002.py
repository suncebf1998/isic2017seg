"""
Jon2022061001


"""
from torch import Tensor
from .unet_parts import *


class MAPCore(nn.Module):
    def __init__(self, channel, ratio=2, drop_ratio=None):
        super().__init__()
        self.conv = DoubleConv(channel, channel)
        self.downup = Down_Up(ratio=ratio)
        if drop_ratio is None:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout2d(drop_ratio)

    def forward(self, x):
        x = self.downup(x) + self.drop(self.conv(x))
        return x

class MultiAveragePool(nn.Module):
    def __init__(self, size, n_channels, n_classes, depth=5, drop_ratio=0.4, bilinear=False):
        super(MultiAveragePool, self).__init__()
        self.size = size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.mapcore = nn.Sequential(
            *[MAPCore(64, 2**i, drop_ratio / depth) if i > 0 else  DoubleConv(64, 64) for i in range(depth-1)]
        )
        self.largerc = DoubleConv(64, 64)
        self.drop = nn.Dropout2d(drop_ratio)
        self.outc = OutConv(64, n_classes)


    # ## Job2022_06_10_01 
    def forward(self, x:Tensor):
        assert x.shape[2] == self.size[0] and x.shape[3] == self.size[1], \
             f"input x's shape ({x.shape[2]}, {x.shape[3]}) must be the same with size {self.size}"
        x = self.inc(x)
        x = self.mapcore(x)
        x = self.largerc(x)
        x = self.drop(x)
        logits = self.outc(x)
        return logits

    
if __name__ == "__main__":
    model = MultiAveragePool((224,224), 3, 1)
    device = torch.device("cuda:0")
    x = torch.randn(32, 3 , 224, 224)
    model = model.to(device)
    x = x.to(device)
    output = model(x)
    print(output.shape, output.dtype)