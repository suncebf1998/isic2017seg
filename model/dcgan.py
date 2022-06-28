from turtle import forward
import torch
from torch import nn
import einops


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ConvUpSample(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2, kernel_size=3, padding=1, batch_eps=0.8, negative_slope=0.2, inplace=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor) if scale_factor != 1 else nn.Identity(),
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_chans, batch_eps),
            nn.LeakyReLU(negative_slope, inplace=inplace)
        )
    
    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels=8, depth=[2,2,2,2], num_head=[24, 12, 6, 3], window_size=(7, 7)):
        super().__init__()
        self.depth = depth
        self.init_size = img_size // 8
        self.l1 = nn.Linear(latent_dim, 256 * self.init_size ** 2)
        self.num_head = num_head
        self.norm1 = nn.BatchNorm2d(256)
        self.conv_blocks = nn.ModuleList()
        self.dim = channels
        self.window_size = window_size
        # for i in 
        for i in range(4):
            decoder_block = nn.ModuleList()
            if i == 0:
                for j in range(depth[i]):
                    block = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256, 0.8),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                    decoder_block.append(block)
                self.conv_blocks.append(decoder_block)
                continue
            for j in range(depth[i]):
                decoder_block.append(ConvUpSample(512 // 2 ** i if j == 0 else 256 // 2 ** i, 256 // 2 ** i, 2 if j ==0 else 1))
                # 256, 256; 256, 128; 128. 64; 64, 32
            self.conv_blocks.append(decoder_block)
           
        self.conv_outs = nn.ModuleList()
        for i in range(4):
            out_block = nn.ModuleList()
            for j in range(depth[i]):
                out_block.append(
                    nn.Sequential(
                        # nn.Conv2d(256 // 2 ** i ,2 * channels * num_head[i], 3, stride=1, padding=1),
                        nn.Conv2d(256 // 2 ** i ,2 * channels * num_head[i], 1, stride=1),
                        nn.Tanh()
                    )
                )
            self.conv_outs.append(out_block)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        outputs = self.decoder(out)
        return outputs

    def decoder(self, x):
        outputs = []
        for i, conv_block, conv_out in zip(range(len(self.num_head)), self.conv_blocks, self.conv_outs):
            output = []
            for block_layer, out_layer in zip(conv_block, conv_out):
                x = block_layer(x)
                qk = out_layer(x) # .flatten(-2,-1) # B, C, N
                qk = einops.rearrange(
                    qk, 
                    "B (n num_head dim) (nH window_size_h) (nW window_size_w) -> n (B nH nW) num_head (window_size_h window_size_w) dim", 
                    n=2, num_head=self.num_head[i], dim=self.dim, 
                    window_size_h=self.window_size[0], window_size_w=self.window_size[1])
                q, k = qk[0], qk[1]
                output.append(q @ k.transpose(-2, -1))
            outputs.append(output[::-1])

        return outputs[::-1]




if __name__ == "__main__":
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
    img_size = 224 // 4
    latent_dim = 100
    # channels = 96
    model = Generator(img_size, latent_dim)
    get_parameter_number(model, True)
    x = torch.randn(32, 100)
    outputs = model(x)
    for output in outputs:
        for o in output:
            print(o.shape)
