import torch
from torch import nn
from utils.data_loading import *
from torch.utils.data import DataLoader

class TimeEmbedding(nn.Module):
    """
    Time embedding for timesteps, dimension same as d_model
    :param max_seq_len: type int, max sequence length of input allowed
    :param d_model: type int, embedding dimension for tokens
    :param device: device to use when storing these encodings
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, T=1000, d_model=256, device='cpu'):
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

class GaussianDiffuser(object):
    def __init__(self, betas=[1e-4, 0.02], T=1000, d_model=256, device='cpu'):
        self.T = T
        self.time_embedding = TimeEmbedding(T, d_model)
        self.betas = torch.linspace(betas[0], betas[1], steps=T)
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        # to align with timestep indices
        self.alphas_bar = torch.cat([torch.tensor([0]), self.alphas_bar])

    def __call__(self, x0):
        batch_size, *_ = x0.shape
        # sample timesteps
        ts = torch.randint(1,self.T+1,size=(batch_size,))
        t_embs = self.time_embedding(ts)
        # sample noise from a normal distribution
        eps = torch.randn_like(x0)
        alphas_bar_t = self.alphas_bar[ts]
        alphas_bar_t_shape = torch.cat([torch.tensor(x0.shape[:-3]), torch.tensor([1]*3)])
        alphas_bar_t = alphas_bar_t.view(tuple(alphas_bar_t_shape))
        xt = torch.sqrt(alphas_bar_t)*x0 + torch.sqrt(1 - alphas_bar_t) * eps
        return xt, eps, t_embs

if __name__ == "__main__":
    gaussian_diffuser = GaussianDiffuser()
    cifar_set = make_cifar_set(diffuser=gaussian_diffuser)
    batch_size = 16
    cifar_loader = DataLoader(cifar_set, batch_size=batch_size, shuffle=True)
    x = next(iter(cifar_loader))
