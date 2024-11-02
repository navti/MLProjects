import torch
from utils.data_loading import *
from torch.utils.data import DataLoader

class GaussianDiffuser(object):
    def __init__(self, betas=[1e-4, 0.02], T=1000, device='cpu'):
        self.T = T
        self.betas = torch.linspace(betas[0], betas[1], steps=T)
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        # to align with timestep indices
        self.alphas_bar = torch.cat([torch.tensor([0]), self.alphas_bar])

    def __call__(self, x0):
        batch_size, *_ = x0.shape
        # sample timesteps
        ts = torch.randint(1,self.T+1,size=(batch_size,))
        # sample noise from a normal distribution
        eps = torch.randn_like(x0)
        alphas_bar_t = self.alphas_bar[ts]
        alphas_bar_t_shape = torch.cat([torch.tensor(x0.shape[:-3]), torch.tensor([1]*3)])
        alphas_bar_t = alphas_bar_t.view(tuple(alphas_bar_t_shape))
        xt = torch.sqrt(alphas_bar_t)*x0 + torch.sqrt(1 - alphas_bar_t) * eps
        return xt, eps, ts

if __name__ == "__main__":
    gaussian_diffuser = GaussianDiffuser()
    cifar_set = make_cifar_set(diffuser=gaussian_diffuser)
    batch_size = 16
    cifar_loader = DataLoader(cifar_set, batch_size=batch_size, shuffle=True)
    x = next(iter(cifar_loader))
