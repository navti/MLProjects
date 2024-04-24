import torch
from torch import nn
import torch.nn.functional as F

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_batchnorm=True, use_activations=True, activation_last=True):
        super(UpConv, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activations:
            relu = nn.ReLU()
            layers.append(relu) if activation_last else layers.insert(1, relu)
        self.upconv = nn.Sequential(*layers)

    def forward(self, X):
        return self.upconv(X)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            UpConv(latent_dim, 512, 4, 2, 1),
            UpConv(512, 256, 4, 2, 1),
            UpConv(256, 128, 4, 2, 1),
            UpConv(128, 64, 4, 2, 1),
            UpConv(64, 3, 4, 2, 1, use_batchnorm=False, use_activations=False),
            nn.Sigmoid()
        )

    def forward(self, latent_vector):
        shape = latent_vector.shape
        arg1 = shape[0]
        if latent_vector.dim() == 1:
            arg1 = 1
        return self.generate(latent_vector.view(arg1,-1,1,1))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_batchnorm=True, use_activations=True, activation_last=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activations:
            relu = nn.ReLU()
            layers.append(relu) if activation_last else layers.insert(1, relu)
        self.conv = nn.Sequential(*layers)

    def forward(self, image):
        return self.conv(image)

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.backbone = nn.Sequential(
            ConvBlock(3, 64, 4, 2, 1),
            ConvBlock(64, 128, 4, 2, 1),
            ConvBlock(128, 256, 4, 2, 1),
            ConvBlock(256, 512, 4, 2, 1),
        )
        self.clf = ConvBlock(512, n_classes, 4, 2, 1)
        self.discriminate = nn.Sequential(
            ConvBlock(n_classes, 1, 3, 1, 1, use_batchnorm=False, use_activations=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        out = self.backbone(image)
        labels = self.clf(out)
        return self.discriminate(labels).view(-1), labels.squeeze()

class GAN(nn.Module):
    def __init__(self, n_classes, latent_dim=100):
        super(GAN, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator(n_classes=n_classes)
    def forward(self, latent_vector):
        image = self.generator(latent_vector)
        is_real, labels = self.discriminator(image)
        return image, is_real, labels

# test model
if __name__ == "__main__":
    latent_dim = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = GAN(latent_dim=latent_dim).to(device=device)
    test_vector = torch.randn(size=(5,latent_dim)).to(device=device)
    gen_out, is_real = test_model(test_vector)
    print(gen_out.shape)
    print(is_real.shape)