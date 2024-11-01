import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['UNet']


class TimeEmbedding(nn.Module):
    """
    Time embedding for timesteps, dimension same as d_model
    :param max_seq_len: type int, max sequence length of input allowed
    :param d_model: type int, embedding dimension for tokens
    :param device: device to use when storing these encodings
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, T, d_model, device='cpu'):
        super(TimeEmbedding, self).__init__()
        self.device = device
        # initialize encoding table with zeros
        self.time_embedding = torch.zeros(size=(T+1, d_model), dtype=torch.float32, device=device)
        # time steps
        time_steps = torch.arange(T+1, dtype=torch.float32, device=device)
        # add a dimension so braodcasting would be possible
        time_steps = time_steps.unsqueeze(dim=1)
        # exponent : 2i
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)
        # sin terms at idx 2i
        stop = d_model//2 # to cover for odd d_model in cosine
        self.time_embedding[:,0::2] = torch.sin(time_steps / (10000 ** (_2i / d_model)))
        # cos terms at idx 2i+1
        self.time_embedding[:,1::2] = torch.cos(time_steps / (10000 ** (_2i[:stop] / d_model)))

    def forward(self, ts):
        """
        forward method to calculate time embedding
        :param ts: type tensor[int]/list[int], input time steps
        :return:
            embeddings: size: batch, T+1, d_model
        """
        time_steps = torch.tensor(ts, device=self.device)
        return self.time_embedding[time_steps, :]


""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activations=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if activations:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool """

    def __init__(self):
        super(Down,self).__init__()
        self.down = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.down(x)


class DownDoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activations=True):
        super(DownDoubleConv, self).__init__()
        self.down_doubleconv = nn.Sequential(
            Down(),
            DoubleConv(in_channels, out_channels, activations=activations)
        )

    def forward(self, x):
        return self.down_doubleconv(x)


class UpDoubleConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activations=True):
        super(UpDoubleConv, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.double_conv = DoubleConv(in_channels, out_channels, in_channels // 2, activations=activations)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.double_conv = DoubleConv(in_channels, out_channels, activations=activations)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, checkpointing=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.checkpointing = checkpointing

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownDoubleConv(64, 128)
        self.down2 = DownDoubleConv(128, 256)
        self.down3 = DownDoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownDoubleConv(512, 1024 // factor)
        self.up1 = UpDoubleConv(1024, 512 // factor, bilinear)
        self.up2 = UpDoubleConv(512, 256 // factor, bilinear)
        self.up3 = UpDoubleConv(256, 128 // factor, bilinear)
        self.up4 = UpDoubleConv(128, 64, bilinear, activations=False)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.checkpointing:
            x1 = checkpoint(self.inc, x, use_reentrant=False)
            x2 = checkpoint(self.down1, x1, use_reentrant=False)
            x3 = checkpoint(self.down2, x2, use_reentrant=False)
            x4 = checkpoint(self.down3, x3, use_reentrant=False)
            x5 = checkpoint(self.down4, x4, use_reentrant=False)
            x = checkpoint(self.up1, x5,x4, use_reentrant=False)
            x = checkpoint(self.up2, x,x3, use_reentrant=False)
            x = checkpoint(self.up3, x,x2, use_reentrant=False)
            x = checkpoint(self.up4, x,x1, use_reentrant=False)
            logits = checkpoint(self.outc, x, use_reentrant=False)
            # probs = checkpoint(self.sigmoid, logits, use_reentrant=False)
            return logits

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits