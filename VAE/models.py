import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

# Conv block used in Resnet model
def conv_block(in_channels, out_channels, pool=False, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
              nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# Transposed 2d convolution
def deconv_block(in_channels, out_channels, kernel_size=2, stride=2, padding=0, activation=True):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
              nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# VAE Encoder: Resnet9
class Encoder(nn.Module):
    def __init__(self, in_channels, nf, latent_dim, device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.device = device
        self.nf = nf
        self.conv1 = conv_block(in_channels, nf)
        self.conv2 = conv_block(nf, 2*nf, pool=True)
        self.res1 = nn.Sequential(conv_block(2*nf, 2*nf), conv_block(2*nf, 2*nf))

        self.conv3 = conv_block(2*nf, 4*nf, pool=True)
        self.conv4 = conv_block(4*nf, 8*nf, pool=True)
        self.res2 = nn.Sequential(conv_block(8*nf, 8*nf), conv_block(8*nf, 8*nf))

        self.conv5 = conv_block(8*nf, 16*nf, pool=True)

        # Linear layer
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(16*nf*2*2, 32*nf),
                                    nn.ReLU(inplace=True))
        # mean and logvar linear layers
        self.mu_linear = nn.Sequential(nn.Linear(32*nf, latent_dim))
        self.logvar_linear = nn.Sequential(nn.Linear(32*nf, latent_dim))

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.conv5(x)
        x = self.linear(x)
        mu = self.mu_linear(x)
        log_var = self.logvar_linear(x)
        return mu, log_var
    
# VAE Decoder : Resnet9
class Decoder(nn.Module):
    def __init__(self, nf, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(latent_dim, 32*nf),
                                     nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(32*nf, 16*nf*2*2),
                                     nn.ReLU(inplace=True))
        self.dconv1 = deconv_block(16*nf, 8*nf, kernel_size=3, stride=1)
        self.res1 = nn.Sequential(conv_block(8*nf, 8*nf), conv_block(8*nf, 8*nf))
        self.dconv2 = deconv_block(8*nf, 4*nf, kernel_size=5, stride=1)
        self.dconv3 = deconv_block(4*nf, 2*nf, kernel_size=9, stride=1)
        self.res2 = nn.Sequential(conv_block(2*nf, 2*nf), conv_block(2*nf, 2*nf))
        self.dconv4 = deconv_block(2*nf, nf, kernel_size=17, stride=1)
        self.dconv5 = conv_block(nf, 3,activation=False)

    def forward(self, z):
        z = self.linear1(z)
        z = self.linear2(z)
        z = z.view(z.shape[0],-1,2,2)
        z = self.dconv1(z)
        z = z + self.res1(z)
        z = self.dconv2(z)
        z = self.dconv3(z)
        z = z + self.res2(z)
        z = self.dconv4(z)
        xhat = self.dconv5(z)
        return xhat

# VAE model : Resnet based
class VAE(nn.Module):
    def __init__(self, in_channels, nf, latent_dim, n_classes=10, device=torch.device('cpu')):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, nf, latent_dim, device)
        self.decoder = Decoder(nf, latent_dim)
        self.classifier = nn.Sequential(nn.Linear(latent_dim, n_classes))
        self.to(device)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        sigma = torch.exp(log_var/2)
        epsilon = self.encoder.N.sample(mu.shape)
        z = mu + sigma*epsilon
        out_targets = self.classifier(z)
        xhat = self.decoder(z)
        return xhat, x, mu, log_var, out_targets
