import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import pathlib
import time
from utils.plotting import *
import os
from utils.data_loading import make_cifar_set
from diffuser import *
from torch.utils.data import DataLoader


""" Parts of the U-Net model """


class ResBlock(nn.Module):
    def __init__(self, in_c, d_model=256, dropout=0.1, Activation=nn.ReLU):
        super().__init__()
        self.activation = nn.Identity() if Activation == None else Activation()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            self.activation,
        )
        self.temb_proj = nn.Sequential(
            nn.Linear(d_model, in_c),
            self.activation,
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=1, padding=1),
            nn.Identity() if Activation == None else nn.BatchNorm2d(in_c),
            nn.Identity() if Activation == None else nn.Dropout(dropout),
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.temb_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        h = h + x
        h = self.activation(h)
        return h


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(
        self, in_c, out_c, mid_c=None, d_model=256, dropout=0.1, Activation=nn.ReLU
    ):
        super(DoubleConv, self).__init__()
        if not mid_c:
            mid_c = out_c
        activation = nn.Identity() if Activation == None else Activation()
        self.temb_proj = nn.Sequential(
            nn.Linear(d_model, out_c),
            activation,
        )
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            activation,
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
            activation,
        )
        self.res = ResBlock(out_c, d_model, dropout=dropout, Activation=Activation)

    def forward(self, x, t_emb):
        t = self.temb_proj(t_emb)[:, :, None, None]
        x = self.double_conv(x) + t
        return self.res(x, t_emb)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        return self.down(x)


class DownDoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_c, out_c, Activation=nn.ReLU):
        super(DownDoubleConv, self).__init__()
        self.down = Down()
        self.double_conv = DoubleConv(in_c, out_c, Activation=Activation)

    def forward(self, x, t_emb):
        x = self.down(x)
        return self.double_conv(x, t_emb)


class UpDoubleConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_c,
        out_c,
        bilinear=False,
        Activation=nn.ReLU,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super(UpDoubleConv, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.double_conv = DoubleConv(in_c, out_c, in_c // 2, Activation=Activation)
        else:
            self.up = nn.ConvTranspose2d(
                in_c,
                in_c // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.double_conv = DoubleConv(
                (out_c + in_c // 2), out_c, Activation=Activation
            )

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x, t_emb)


class UNet(nn.Module):
    """
    args list: n_channels=3, n_classes=10, d_model=256, nf=8, bilinear=False

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
        self.bilinear = (
            kwargs.get("bilinear")
            if "bilinear" in kwargs
            else args[5] if len(args) >= 5 else False
        )

        d_model = self.d_model
        n_channels = self.n_channels
        nf = self.nf

        self.head = DoubleConv(n_channels, nf)
        self.down1 = DownDoubleConv(nf, 2 * nf)
        self.down2 = DownDoubleConv(2 * nf, 4 * nf)
        self.down3 = DownDoubleConv(4 * nf, 8 * nf)
        self.down4 = DownDoubleConv(8 * nf, 16 * nf)
        self.down5 = DownDoubleConv(16 * nf, d_model)

        self.up1 = UpDoubleConv(d_model, 16 * nf)
        self.up2 = UpDoubleConv(16 * nf, 8 * nf)
        self.up3 = UpDoubleConv(8 * nf, 4 * nf)
        self.up4 = UpDoubleConv(4 * nf, 2 * nf)
        self.up5 = UpDoubleConv(2 * nf, nf)
        self.tail = UpDoubleConv(nf, n_channels, False, None, 3, 1, 1)

    def forward(self, x, t_emb):
        x1 = self.head(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)
        x6 = self.down5(x5, t_emb)

        y = self.up1(x6, x5, t_emb)
        y = self.up2(y, x4, t_emb)
        y = self.up3(y, x3, t_emb)
        y = self.up4(y, x2, t_emb)
        y = self.up5(y, x1, t_emb)
        y = self.tail(y, x, t_emb)

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
    test_model = UNet(n_channels=n_channels, n_classes=n_classes, nf=nf)
    xt, eps, t_embs, ts, labels = next(iter(train_loader))
    out = test_model(xt, t_embs)
    print(f"{out.shape}")
