import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['UNet']

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