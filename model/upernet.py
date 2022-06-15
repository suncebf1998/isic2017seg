import torch
from torch import nn
from .upernetparts import UPerHead
from .unet_parts import DoubleConv, Down


class UPerNet(nn.Module):
    def __init__(self, n_channels, num_classes, depth=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = num_classes
        self.inc = DoubleConv(n_channels, 64) #
        self.down_encoder = nn.ModuleList()
        out_channels = [64,]
        for in_factor in range(depth):
            in_channel = 64 * 2 ** in_factor
            out_channels.append(in_channel * 2)
            down = Down(in_channel, out_channels[-1])
            self.down_encoder.append(down)
            # 
        
        self.uperhead = UPerHead(
            in_channels=out_channels, 
            in_index=tuple(range(depth+1)),
            channels=int(out_channels[0] / 2), 
            conv_cfg="Conv2d", norm_cfg="BN", act_cfg="ReLU",
            num_classes=num_classes, dropout_ratio=0.2)

    def encoder_forward(self, x):
        output = self.inc(x)
        outputs = [output, ]
        # output = output.clone()
        for down_layer in self.down_encoder:
            output = down_layer(output)
            outputs.append(output)
        return tuple(outputs)
    
    

    def forward(self, x):
        
        return self.uperhead(self.encoder_forward(x))


if __name__ == "__main__":
    device = "cuda:0"
    data = torch.randn(16, 3, 224, 224).to(device)
    upernet = UPerNet(3, 1).to(device)


    def get_parameter_number(model: torch.nn.Module, printable:bool=False) -> dict:
        """
        stat the total param num and the num of trainable
        model: the model to be evaluated.
        ret: the dict of "Total" and "Trainable"
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info = {'Total': total_num, 'Trainable': trainable_num}
        if printable:
            for key, value in model_info.items():
                print(key, value, sep="\t")
        return model_info
    
    print(upernet)
    get_parameter_number(upernet, True)
    outputs = upernet.encoder_forward(data)
    print("----outputs info----")
    print(len(outputs))
    for output in outputs:
        print(output.shape)
    print("----outputs info----")
    print("----final output----")
    final_output = upernet.uperhead(outputs)
    print(final_output.shape)
    print(final_output.dtype)



    