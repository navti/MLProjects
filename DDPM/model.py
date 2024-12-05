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


def init_weights(modules):
    for m in modules:
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 32 if channels % 32 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        # self.norm = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self._init_weights()

    def _init_weights(self):
        init_weights([self.qkv])
        init_weights([self.proj_out])

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        attn = torch.bmm(q, k) / (C**0.5)  # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(out)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim=256, dropout=0.1, Activation=nn.ReLU):
        super().__init__()
        groups = 32 if in_c % 32 == 0 else 1
        activation = nn.Identity() if Activation == None else Activation()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_c),
            # nn.BatchNorm2d(in_c),
            activation,
            nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.Linear(t_dim, out_c),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_c),
            # nn.BatchNorm2d(out_c),
            activation,
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
        )
        # skip connection with conv if in_c is not same as out_c
        self.skip_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        init_weights(self.block1)
        init_weights(self.block2)
        init_weights(self.temb_proj)
        init_weights([self.skip_conv])

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.temb_proj(t_emb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.skip_conv(x)
        return h


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self, in_c):
        super(Down, self).__init__()
        self.down = nn.Conv2d(in_c, in_c, 4, 2, 1)
        self._init_weights()

    def _init_weights(self):
        init_weights([self.down])

    def forward(self, x):
        return self.down(x)


class ResDown(nn.Module):
    def __init__(self, in_c, out_c, t_dim=256, dropout=0.1, Activation=nn.ReLU):
        super(ResDown, self).__init__()
        self.res1 = ResBlock(in_c, out_c, t_dim, dropout, Activation)
        self.attn1 = AttentionBlock(out_c)
        self.res2 = ResBlock(out_c, out_c, t_dim, dropout, Activation)
        self.attn2 = AttentionBlock(out_c)
        self.down = Down(out_c)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        if x.shape[-1] <= 8:
            x = self.attn1(x)
        x = self.res2(x, t_emb)
        if x.shape[-1] <= 8:
            x = self.attn2(x)
        x = self.down(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, t_dim=256, dropout=0.1, Activation=nn.ReLU):
        super(Bottleneck, self).__init__()
        self.res1 = ResBlock(in_c, out_c, t_dim, dropout, Activation)
        self.res2 = ResBlock(out_c, out_c, t_dim, dropout, Activation)
        self.attn1 = AttentionBlock(out_c)
        self.attn2 = AttentionBlock(out_c)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn1(x)
        x = self.res2(x, t_emb)
        x = self.attn2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_c, out_c, Activation=nn.ReLU):
        super(Up, self).__init__()
        groups = 32 if out_c % 32 == 0 else 1
        activation = nn.Identity() if Activation == None else Activation()
        self.up_sample = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ConvTranspose2d(in_c, in_c, 4, 2, 1),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            # nn.BatchNorm2d(out_c),
            nn.GroupNorm(groups, out_c),
            activation,
        )
        self._init_weights()

    def _init_weights(self):
        init_weights(self.up_sample)

    def forward(self, x):
        x = self.up_sample(x)
        return x


class UpRes(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_c, out_c, t_dim=256, dropout=0.1, Activation=nn.ReLU):
        super(UpRes, self).__init__()
        self.up = Up(in_c, out_c, Activation=Activation)
        self.res1 = ResBlock(in_c + out_c, out_c, t_dim, dropout, Activation)
        self.attn1 = AttentionBlock(out_c)
        self.res2 = ResBlock(out_c, out_c, t_dim, dropout, Activation)
        self.attn2 = AttentionBlock(out_c)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.res1(x, t_emb)
        if x.shape[-1] <= 8:
            x = self.attn1(x)
        x = self.res2(x, t_emb)
        if x.shape[-1] <= 8:
            x = self.attn2(x)
        return x


class HeadConv(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super(HeadConv, self).__init__()
        groups = 32 if in_c % 32 == 0 else 1
        self.temb_proj = nn.Sequential(
            nn.Linear(t_dim, out_c),
        )
        self.conv = nn.Sequential(
            # nn.GroupNorm(groups, in_c),
            # nn.ReLU(),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        init_weights(self.temb_proj)
        init_weights(self.conv)

    def forward(self, x, t_emb):
        x = self.conv(x)
        x = x + self.temb_proj(t_emb)[:, :, None, None]
        return x


class FinalConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(FinalConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            # nn.BatchNorm2d(mid_c),
            # nn.ReLU(),
            # nn.Conv2d(mid_c, out_c, 3, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    args list: n_channels=3, n_classes=10, t_dim=256, nf=8, bilinear=False

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
        self.t_dim = (
            kwargs.get("t_dim")
            if "t_dim" in kwargs
            else args[2] if len(args) >= 3 else 256
        )
        self.nf = (
            kwargs.get("nf") if "nf" in kwargs else args[3] if len(args) >= 4 else 16
        )
        self.bilinear = (
            kwargs.get("bilinear")
            if "bilinear" in kwargs
            else args[5] if len(args) >= 5 else False
        )

        t_dim = self.t_dim
        n_channels = self.n_channels
        nf = self.nf

        # self.head = nn.Conv2d(n_channels, nf, 3, 1, 1)  # 32x32
        self.head = HeadConv(n_channels, nf, t_dim=t_dim)  # 32x32
        self.enc1 = ResDown(nf, 2 * nf, t_dim=t_dim)  # 16x16
        self.enc2 = ResDown(2 * nf, 4 * nf, t_dim=t_dim)  # 8x8
        self.enc3 = ResDown(4 * nf, 4 * nf, t_dim=t_dim)  # 4x4
        self.bottleneck = Bottleneck(4 * nf, 4 * nf, t_dim=t_dim)  # 2x2
        self.dec3 = UpRes(4 * nf, 4 * nf, t_dim=t_dim)  # 4x4
        self.dec2 = UpRes(4 * nf, 2 * nf, t_dim=t_dim)  # 8x8
        self.dec1 = UpRes(2 * nf, nf, t_dim=t_dim)  # 16x16
        self.tail = FinalConv(nf, n_channels)  # 32x32

    def forward(self, x, t_emb):
        x = self.head(x, t_emb)
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(x1, t_emb)
        x3 = self.enc3(x2, t_emb)
        x4 = self.bottleneck(x3, t_emb)
        y = self.dec3(x4, x3, t_emb)
        y = self.dec2(y, x2, t_emb)
        y = self.dec1(y, x1, t_emb)
        y = self.tail(y)
        return y


def save_checkpoint(
    model,
    optimizer,
    losses,
    lr_schedule,
    checkpoint_dir,
    scheduler=None,
    filename="checkpoint",
):
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_chk_path = checkpoint_dir + "/" + filename + ".pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "lr_schedule": lr_schedule,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, save_chk_path)


def load_checkpoint(model, filepath, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    losses = checkpoint["losses"]
    lr_schedule = checkpoint["lr_schedule"]
    return losses, lr_schedule


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
    gaussian_diffuser = GaussianDiffuser(d_model=512)
    trainset, _ = make_cifar_set(data_dir=cifar_data_dir, diffuser=gaussian_diffuser)
    train_loader = DataLoader(trainset, batch_size=16)
    n_channels = 3
    n_classes = 10
    nf = 32
    test_model = UNet(n_channels=n_channels, n_classes=n_classes, nf=nf, t_dim=512)
    xt, eps, t_embs, ts, labels = next(iter(train_loader))
    out = test_model(xt, t_embs)
    print(f"{out.shape}")
