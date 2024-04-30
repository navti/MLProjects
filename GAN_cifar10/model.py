import torch
from torch import nn

def weights_init(neural_net):
    """
    custom weights init function
    :param neural_net: model whose weights should be initialized
    :returns: None
    """
    classname = neural_net.__class__.__name__
    if classname.find('Conv') != -1:
        neural_net.weight.data.normal_(0, 2e-2)
    elif classname.find('BatchNorm') != -1:
        neural_net.weight.data.normal_(1, 2e-2)
        neural_net.bias.data.fill_(0)

class UpConv(nn.Module):
    """
    transposed convolution block
    :param in_channels: input channels
    :param out_channels: out channels/no. of filters
    :param kernel_size: size of filters
    :param stride: filter stride
    :param padding: filter padding
    :param use_batchnorm: boolean, use batch normalization
    :param use_activation: boolean, use leaky relu activations
    :param activation_last: boolean, put activations as last layer in the block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_batchnorm=True, use_activations=True, activation_last=True):
        super(UpConv, self).__init__()
        # layers in the transposed conv block
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batchnorm:
            # add batch norm layer
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activations:
            # add activations before/after batch norm depending on activation_last
            relu = nn.LeakyReLU(0.2) # nn.ReLU()
            layers.append(relu) if activation_last else layers.insert(1, relu)
        # compile layers in a sequential object
        self.upconv = nn.Sequential(*layers)

    # forward pass
    def forward(self, X):
        return self.upconv(X)

class Generator(nn.Module):
    """
    generator class to generate images from noise samples
    :param device: device that hosts model
    :param latent_dim: latent vector dimension
    """
    def __init__(self, device, latent_dim=100):
        super(Generator, self).__init__()
        # latent dimension
        self.latent_dim = latent_dim
        self.device = device
        # mean and log variance for noise vector
        self.mean = nn.parameter.Parameter(
            data=torch.zeros(size=(1, latent_dim), dtype=torch.float32, requires_grad=True)
        )
        self.log_var = nn.parameter.Parameter(
            data=torch.ones(size=(1, latent_dim), dtype=torch.float32, requires_grad=True)
        )
        # Normal object to sample from normal distribution
        self.N = torch.distributions.Normal(loc=0.0, scale=0.5)
        # move loc and scale to device, aid sampling directly on device
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        # generate layers
        self.generate = nn.Sequential(
            UpConv(latent_dim, 512, 4, 2, 1),
            UpConv(512, 256, 4, 2, 1),
            UpConv(256, 128, 4, 2, 1),
            UpConv(128, 64, 4, 2, 1),
            UpConv(64, 32, 4, 2, 1),
            UpConv(32, 3, 3, 1, 1, use_batchnorm=True, use_activations=False),
            nn.Sigmoid()
        )
        # fixed noise
        self.eval_noise = self.sample_latent_vectors(n_samples=50)
        self.eval_noise = self.eval_noise.to(device)

    def sample_latent_vectors(self, n_samples):
        """
        sample noise vector
        :param n_samples: no. of vectors to sample
        :returns: latent vector tensor of size n_samples x latent_dim
        """
        epsilon = self.N.sample(sample_shape=(n_samples, self.latent_dim))
        sigma = torch.exp(self.log_var/2).expand(n_samples, self.latent_dim).to(self.device)
        u = self.mean.expand(n_samples, self.latent_dim).to(self.device)
        latent_vectors = u + sigma*epsilon
        return latent_vectors

    # forward pass
    def forward(self, latent_vector):
        shape = latent_vector.shape
        arg1 = shape[0]
        if latent_vector.dim() == 1:
            arg1 = 1
        return self.generate(latent_vector.view(arg1,-1,1,1))

class ConvBlock(nn.Module):
    """
    Convolution block to used by the discriminator
    :param in_channels: input channels
    :param out_channels: out channels/no. of filters
    :param kernel_size: size of filters
    :param stride: filter stride
    :param padding: filter padding
    :param use_batchnorm: boolean, use batch normalization
    :param use_activation: boolean, use leaky relu activations
    :param activation_last: boolean, put activations as last layer in the block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_batchnorm=True, use_activations=True, activation_last=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activations:
            relu = nn.LeakyReLU(0.2) # nn.ReLU()
            layers.append(relu) if activation_last else layers.insert(1, relu)
        self.conv = nn.Sequential(*layers)

    def forward(self, image):
        return self.conv(image)

class Discriminator(nn.Module):
    """
    Discriminator class to differentiate between real and fake images
    :param n_classes: no. of classes in the dataset
    """
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.backbone = nn.Sequential(
            ConvBlock(3, 64, 4, 2, 1),
            ConvBlock(64, 128, 4, 2, 1),
            ConvBlock(128, 256, 4, 2, 1),
            ConvBlock(256, 512, 4, 2, 1),
            ConvBlock(512, 256, 4, 2, 1),
        )
        # classification layer
        self.clf = ConvBlock(256, n_classes, 3, 1, 1, use_batchnorm=False, use_activations=False)
        self.discriminate = nn.Sequential(
            ConvBlock(256, 1, 3, 1, 1, use_batchnorm=False, use_activations=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        out = self.backbone(image)
        labels = self.clf(out)
        return self.discriminate(out).view(-1), labels.squeeze()

class GAN(nn.Module):
    """
    Generative Adversarial Network, combines gnerator and discriminator
    :param n_classes: no. of classes in the image dataset
    :param device: device to train the model on
    :param latent_dim: dimesion of the latent noise vector
    """
    def __init__(self, n_classes, device, latent_dim=100):
        super(GAN, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.generator = Generator(device=device, latent_dim=latent_dim)
        self.discriminator = Discriminator(n_classes=n_classes)
    def forward(self, latent_vector):
        image = self.generator(latent_vector)
        is_real, labels = self.discriminator(image)
        return image, is_real, labels

# test model
if __name__ == "__main__":
    """
    test function to validate model
    """
    latent_dim = 100
    n_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = GAN(n_classes=n_classes, device=device, latent_dim=latent_dim).to(device=device)
    # test_vector = torch.randn(size=(5,latent_dim)).to(device=device)
    test_vector = test_model.generator.sample_latent_vectors(n_samples=5)
    gen_out, is_real, clf_labels = test_model(test_vector)
    print(gen_out.shape)
    print(is_real.shape)