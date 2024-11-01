import torch

class GaussianDiffuser:
    def __init__(self, betas=[1e-4, 0.02], T=1000, device='cpu'):
        self.T = T
        self.betas = torch.linspace(betas[0], betas[1], steps=T)
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        # to align with timestep indices
        self.alphas_bar = torch.cat([torch.tensor([0]), self.alphas_bar])

    def diffuse(self, x0):
        batch_size, *_ = x0.shape
        # sample timesteps
        ts = torch.randint(1,self.T+1,size=(batch_size,))
        # sample noise from a normal distribution
        eps = torch.randn_like(x0)
        alphas_bar_t = self.alphas_bar[ts]
        alphas_bar_t_shape = torch.cat([torch.tensor(x0.shape[:-3]), torch.tensor([1]*3)])
        alphas_bar_t = alphas_bar_t.view(tuple(alphas_bar_t_shape))
        xt = torch.sqrt(alphas_bar_t)*x0 + torch.sqrt(1 - alphas_bar_t) * eps
        return xt