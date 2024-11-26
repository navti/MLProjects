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

    def __init__(self, T=1000, d_model=256, device="cpu"):
        super(TimeEmbedding, self).__init__()
        self.device = device
        # initialize encoding table with zeros
        self.time_embedding = torch.zeros(
            size=(T + 1, d_model), dtype=torch.float32, device=device
        )
        # time steps
        time_steps = torch.arange(T + 1, dtype=torch.float32, device=device)
        # add a dimension so braodcasting would be possible
        time_steps = time_steps.unsqueeze(dim=1)
        # exponent : 2i
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)
        # sin terms at idx 2i
        stop = d_model // 2  # to cover for odd d_model in cosine
        self.time_embedding[:, 0::2] = torch.sin(
            time_steps / (10000 ** (_2i / d_model)),
        ).to(device)
        # cos terms at idx 2i+1
        self.time_embedding[:, 1::2] = torch.cos(
            time_steps / (10000 ** (_2i[:stop] / d_model))
        ).to(device)

    def forward(self, time_steps):
        """
        forward method to calculate time embedding
        :param time_steps: type tensor[int]/list[int], input time steps
        :return:
            embeddings: size: batch, T+1, d_model
        """
        return self.time_embedding[time_steps, :]


class GaussianDiffuser(object):
    def __init__(self, betas=[1e-4, 0.02], T=1000, d_model=256):
        self.T = T
        self.time_embedding = TimeEmbedding(T, d_model)
        self.betas = torch.linspace(betas[0], betas[1], steps=T)
        # to align with timestep indices
        self.betas = torch.cat([torch.tensor([0]), self.betas])
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)

    def __call__(self, x0):
        return self.diffuse(x0)

    def diffuse(self, x0):
        batch_size, *_ = x0.shape
        # sample timesteps
        ts = torch.randint(1, self.T + 1, size=(batch_size,))
        t_embs = self.time_embedding(ts)
        # sample noise from a normal distribution
        eps = torch.randn_like(x0)
        alphas_bar_t = self.alphas_bar[ts]
        alphas_bar_t_shape = torch.cat(
            [torch.tensor(x0.shape[:-3]), torch.tensor([1] * 3)]
        )
        alphas_bar_t = alphas_bar_t.view(tuple(alphas_bar_t_shape))
        xt = torch.sqrt(alphas_bar_t) * x0 + torch.sqrt(1 - alphas_bar_t) * eps
        return xt, eps, t_embs, ts

    def sample(self, model, size, device="cpu"):
        # size: BCHW
        model.eval()
        with torch.no_grad():
            batch_size = size[0]
            xt = torch.randn(size=size)
            ts = torch.tensor(list(range(self.T, 0, -1)))
            for t in ts:
                t_emb = self.time_embedding(t)
                t_emb = t_emb.expand(batch_size, -1)
                xt = xt.to(device)
                t_emb = t_emb.to(device)
                eps = model(xt, t_emb)
                factor = self.betas[t] / torch.sqrt(1 - self.alphas_bar[t])
                mean_t_bar = (xt - eps * factor) / torch.sqrt(self.alphas[t])
                sigma_t = torch.sqrt(self.betas[t])
                z = torch.randn_like(mean_t_bar)
                if t == 1:
                    x0 = torch.clamp(mean_t_bar, min=-1.0, max=1.0)
                    x0 = (x0 + 1) / 2
                    return x0
                xt_1 = mean_t_bar + sigma_t * z
                # clear gpu tensors
                del xt, t_emb, eps, mean_t_bar, z
                xt = xt_1.detach().clone()
                del xt_1
                torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussian_diffuser = GaussianDiffuser(device=device)
    cifar_set = make_cifar_set(diffuser=gaussian_diffuser)
    batch_size = 16
    cifar_loader = DataLoader(cifar_set, batch_size=batch_size, shuffle=True)
    x = next(iter(cifar_loader))
