import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import pathlib
import time

import os
from utils.data_loading import make_cifar_set
from diffuser import GaussianDiffuser
from torch.utils.data import DataLoader


""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activations=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if activations:
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                # nn.BatchNorm2d(mid_channels),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                # nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        return self.down(x)


class DownDoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activations=True):
        super(DownDoubleConv, self).__init__()
        self.down_doubleconv = nn.Sequential(
            Down(), DoubleConv(in_channels, out_channels, activations=activations)
        )

    def forward(self, x):
        return self.down_doubleconv(x)


class UpDoubleConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=False,
        activations=True,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super(UpDoubleConv, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.double_conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, activations=activations
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.double_conv = DoubleConv(
                in_channels, out_channels, activations=activations
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    args list: n_channels=3, n_classes=10, d_model=256, nf=8, checkpointing=False, bilinear=False

    """

    def __init__(self, *args, **kwargs):
        super(UNet, self).__init__()
        self.n_channels = (
            kwargs.get("n_channels")
            if "n_channels" in kwargs
            else args[0] if len(args) >= 1 else 3
        )
        self.n_classes = (
            kwargs.get("n_classes")
            if "n_classes" in kwargs
            else args[1] if len(args) >= 2 else 10
        )
        self.d_model = (
            kwargs.get("d_model")
            if "d_model" in kwargs
            else args[2] if len(args) >= 3 else 256
        )
        self.nf = (
            kwargs.get("nf") if "nf" in kwargs else args[3] if len(args) >= 4 else 8
        )
        self.checkpointing = (
            kwargs.get("checkpointing")
            if "checkpointing" in kwargs
            else args[4] if len(args) >= 5 else True
        )
        self.bilinear = (
            kwargs.get("bilinear")
            if "bilinear" in kwargs
            else args[5] if len(args) >= 5 else False
        )

        d_model = self.d_model
        n_channels = self.n_channels
        nf = self.nf

        self.time_proj1 = nn.Linear(d_model, 32 * 32)
        self.time_proj2 = nn.Linear(d_model, 16 * 16)
        self.time_proj3 = nn.Linear(d_model, 8 * 8)
        self.time_proj4 = nn.Linear(d_model, 4 * 4)
        self.time_proj5 = nn.Linear(d_model, 2 * 2)

        self.inc = DoubleConv(n_channels, nf)
        self.down1 = DownDoubleConv(nf, 2 * nf)
        self.down2 = DownDoubleConv(2 * nf, 4 * nf)
        self.down3 = DownDoubleConv(4 * nf, 8 * nf)
        self.down4 = DownDoubleConv(8 * nf, 16 * nf)
        self.down5 = DownDoubleConv(16 * nf, 32 * nf)

        self.up1 = UpDoubleConv(32 * nf, 16 * nf)
        self.up2 = UpDoubleConv(16 * nf, 8 * nf)
        self.up3 = UpDoubleConv(8 * nf, 4 * nf)
        self.up4 = UpDoubleConv(4 * nf, 2 * nf)
        self.up5 = UpDoubleConv(2 * nf, nf)
        self.double_conv1 = DoubleConv(n_channels, n_channels + 1)
        self.up6 = UpDoubleConv(
            nf, n_channels, activations=False, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, t_emb):
        B = x.shape[0]
        if self.checkpointing:
            x1 = checkpoint(self.inc, x, use_reentrant=False)
            t1 = checkpoint(self.time_proj1, t_emb, use_reentrant=False)
            x1 = x1 + t1.view(size=(B, 1, *x1.shape[-2:]))

            x2 = checkpoint(self.down1, x1, use_reentrant=False)
            t2 = checkpoint(self.time_proj2, t_emb, use_reentrant=False)
            x2 = x2 + t2.view(size=(B, 1, *x2.shape[-2:]))

            x3 = checkpoint(self.down2, x2, use_reentrant=False)
            t3 = checkpoint(self.time_proj3, t_emb, use_reentrant=False)
            x3 = x3 + t3.view(size=(B, 1, *x3.shape[-2:]))

            x4 = checkpoint(self.down3, x3, use_reentrant=False)
            t4 = checkpoint(self.time_proj4, t_emb, use_reentrant=False)
            x4 = x4 + t4.view(size=(B, 1, *x4.shape[-2:]))

            x5 = checkpoint(self.down4, x4, use_reentrant=False)
            t5 = checkpoint(self.time_proj5, t_emb, use_reentrant=False)
            x5 = x5 + t5.view(size=(B, 1, *x5.shape[-2:]))

            x6 = checkpoint(self.down5, x5, use_reentrant=False)
            x6 = x6 + t_emb[:, :, None, None]

            y = checkpoint(self.up1, x6, x5, use_reentrant=False)
            y = y + t5.view(size=(B, 1, *y.shape[-2:]))

            y = checkpoint(self.up2, y, x4, use_reentrant=False)
            y = y + t4.view(size=(B, 1, *y.shape[-2:]))

            y = checkpoint(self.up3, y, x3, use_reentrant=False)
            y = y + t3.view(size=(B, 1, *y.shape[-2:]))

            y = checkpoint(self.up4, y, x2, use_reentrant=False)
            y = y + t2.view(size=(B, 1, *y.shape[-2:]))

            y = checkpoint(self.up5, y, x1, use_reentrant=False)
            y = y + t1.view(size=(B, 1, *y.shape[-2:]))

            # 3 ch -> 4 ch, so they can be concatenated to form 8 ch
            x = checkpoint(self.double_conv1, x, use_reentrant=False)

            y = checkpoint(self.up6, y, x, use_reentrant=False)

            return y

        x1 = self.inc(x)
        t1 = self.time_proj1(t_emb)
        x1 = x1 + t1.view(size=(B, 1, *x1.shape[-2:]))

        x2 = self.down1(x1)
        t2 = self.time_proj1(t_emb)
        x2 = x2 + t2.view(size=(B, 1, *x2.shape[-2:]))

        x3 = self.down2(x2)
        t3 = self.time_proj1(t_emb)
        x3 = x3 + t3.view(size=(B, 1, *x3.shape[-2:]))

        x4 = self.down3(x3)
        t4 = self.time_proj1(t_emb)
        x4 = x4 + t4.view(size=(B, 1, *x4.shape[-2:]))

        x5 = self.down4(x4)
        t5 = self.time_proj1(t_emb)
        x5 = x5 + t5.view(size=(B, 1, *x5.shape[-2:]))

        x6 = self.down4(x5)
        x6 = x6 + t_emb[:, :, None, None]

        y = self.up1(x6, x5)
        y = y + t5.view(size=(B, 1, *y.shape[-2:]))

        y = self.up2(y, x4)
        y = y + t4.view(size=(B, 1, *y.shape[-2:]))

        y = self.up3(y, x3)
        y = y + t3.view(size=(B, 1, *y.shape[-2:]))

        y = self.up4(y, x2)
        y = y + t2.view(size=(B, 1, *y.shape[-2:]))

        y = self.up5(y, x1)
        y = y + t1.view(size=(B, 1, *y.shape[-2:]))

        x = self.double_conv1(x)

        y = self.up6(y, x)

        return y


def save_model(model, models_dir, name=None):
    """
    Save model to disk
    :param model: model to be saved
    :param models_dir: directory where the model should be saved
    :param name: model file name
    :return: None
    """
    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_model_path = models_dir + "/ddpm-" + timestr + ".pth"
    else:
        save_model_path = models_dir + "/" + name + ".pth"
    try:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved at: {save_model_path}")
    except OSError as e:
        print("Failed to save model.")
        print(f"{e.strerror}: {e.filename}")


# load model
def load_model(model_path, *args, **kwargs):
    """
    Load model from disk
    :param model_path: model file path
    :param args: args for model to initialize
    :param kwargs: keyword arguments for model to initialize
    :return:
        model: loaded model
    """
    model = UNet(*args, **kwargs)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except OSError as e:
        print(f"{e.strerror}: {e.filename}")
        return None
    return model


if __name__ == "__main__":
    cifar_data_dir = os.path.abspath("./data")
    gaussian_diffuser = GaussianDiffuser()
    trainset = make_cifar_set(data_dir=cifar_data_dir, diffuser=gaussian_diffuser)
    train_loader = DataLoader(trainset, batch_size=16)
    n_channels = 3
    n_classes = 10
    nf = 8
    test_model = UNet(
        n_channels=n_channels, n_classes=n_classes, nf=nf, checkpointing=True
    )
    xt, eps, t_embs, labels = next(iter(train_loader))
    out = test_model(xt, t_embs)
    print(f"{out.shape}")
