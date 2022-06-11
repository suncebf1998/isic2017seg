"""
Jon2022061001


"""
from torch import Tensor
from .unet_parts import *


class MultiAveragePool(nn.Module):
    def __init__(self, size, n_channels, n_classes, depth=5, drop_ratio=0.4, bilinear=False):
        super(MultiAveragePool, self).__init__()
        self.size = size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.inc = DoubleConv(n_channels, 64)
        self.incs = nn.Sequential(
            *[DoubleConv(64, 64) if i > 0 else  DoubleConv(n_channels, 64) for i in range(depth-1)]
        )
        # 64, 128, 256, 512, 1024
        self.incsmaller = DoubleConv(64, 8)
        self.drop = nn.Dropout2d(drop_ratio)
        self.downups = nn.ModuleList()
        for i in range(3):
            self.downups.append(Down_Up(2*2**i)) # 2 4 8
        self.incout = DoubleConv(8, 64)
        self.outc = OutConv(64, n_classes)


    # ## Job2022_06_10_01 
    def forward(self, x:Tensor):
        assert x.shape[2] == self.size[0] and x.shape[3] == self.size[1], \
             f"input x's shape ({x.shape[2]}, {x.shape[3]}) must be the same with size {self.size}"
        x = self.incs(x)
        x = self.incsmaller(x)
        x = self.drop(x)
        xs = x.clone()
        # xs = [x, ]
        for layer in self.downups:
            x_mid = layer(x)
            # xs.append(x_mid)
            xs += x_mid
        # x = sum(xs)
        x_before_out = self.incout(xs)
        logits = self.outc(x_before_out)
        return logits

    
if __name__ == "__main__":
    model = MultiAveragePool((224,224), 3, 1)
    device = torch.device("cuda:0")
    x = torch.randn(32, 3 , 224, 224)
    model = model.to(device)
    x = x.to(device)
    output = model(x)
    print(output.shape, output.dtype)