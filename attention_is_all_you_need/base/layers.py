import torch
from torch import nn
from embeddings import InputEmbedding

class LayerNorm(nn.Module):
    """
    Layer normalization
    mean and variance are calculated per token but the learnable params are per
    embedding dimension.
    :param d_model: embedding dimension of the model
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, device='cpu'):
        super(LayerNorm, self).__init__()
        self.device = device
        # beta and gamma are learnable parameters for layer norm
        self.beta = nn.Parameter(torch.ones(d_model)).to(device)
        self.gamma = nn.Parameter(torch.ones(d_model)).to(device)
        self.eps = 1e-5

    def forward(self, x):
        x = x.to(self.device)
        # mean across last dimension, the embedding dimension
        mean = x.mean(-1, keepdim=True)
        # variance across last dimension
        var = x.var(-1, keepdim=True)
        z = (x - mean)/torch.sqrt(var + self.eps)
        z = self.gamma * z + self.beta
        return z

class FeedForward(nn.Module):
    """
    feed forward layer, stack of linear layers
    :param d_model: type int, embedding dimension
    :param hidden: type int, hidden dimension
    :param drop_prob: type float, dropout probability
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, hidden=None, drop_prob=0.1, device='cpu'):
        super(FeedForward, self).__init__()
        self.device = device
        if hidden is None:
            hidden = d_model // 2
        self.linear1 = nn.Linear(d_model, hidden).to(device)
        self.linear2 = nn.Linear(hidden, hidden).to(device)
        self.linear3 = nn.Linear(hidden, d_model).to(device)
        # use ReLU activation
        self.relu = nn.ReLU().to(device)
        self.dropout = nn.Dropout(p=drop_prob).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        return self.linear3(x)

class MultiHeadSelfAttention(nn.Module):
    """
    multi head self attention for encoder/decoder
    :param d_model: type int, token embedding dimension
    :param num_attn_heads: type int, no. of attention heads
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, num_attn_heads, device='cpu'):
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linear = nn.Linear(d_model, d_model, bias=False).to(device)
        self.self_attn_heads = nn.ModuleList([SelfAttentionHead(d_model, num_attn_heads, device) for _ in range(num_attn_heads)])

    def forward(self, x, mask=None, cross_attn=False, enc=None):
        """
        multi head attention call
        :param x: type tensor, token embeddings
        :param mask: attention mask
        :param cross_attn: type bool, if cross attention should be used, applicable for decoder
        :param enc: type tensor, output from encoder, used by decoder for cross attention
        :return:
            out: type tensor, concatenated outputs from attention heads
        """
        # x token embeddings, batch_size x length x d_model
        out = []
        for self_attn_head in self.self_attn_heads:
            out.append(self_attn_head(x, mask, cross_attn, enc))
        # concatenate self attn head outputs
        z = torch.cat(out, dim=-1).to(self.device)
        return self.linear(z)

class SelfAttentionHead(nn.Module):
    """
    self attention for encoder/decoder
    :param d_model: type int, token embedding dimension
    :param num_attn_heads: type int, no. of attention heads
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, num_attn_heads, device='cpu'):
        super(SelfAttentionHead, self).__init__()
        self.device = device
        self.head_dim = d_model//num_attn_heads
        assert self.head_dim * num_attn_heads == d_model, "model dimension not a multiple of attn head dimension."
        self.qkv = nn.Linear(d_model, 3*self.head_dim, bias=False).to(device)
        self.dot_product = ScaledDotProductAttention(device=device)
    
    def forward(self, x, mask=None, cross_attn=False, enc=None):
        """
        self attention call
        :param x: type tensor, token embeddings
        :param mask: attention mask
        :param cross_attn: type bool, if cross attention should be used, applicable for decoder
        :param enc: type tensor, output from encoder, used by decoder for cross attention
        :return:
            out: type tensor, scaled dot product
        """
        x = x.to(self.device)
        # x token embeddings, batch_size x length x d_model
        self_qkv = self.qkv(x)
        q = self_qkv[:, :, :self.head_dim]
        k = self_qkv[:, :, self.head_dim:2*self.head_dim]
        v = self_qkv[:, :, 2*self.head_dim:]
        # cross attention used in decoder
        if cross_attn:
            assert enc is not None, "enc input needed for encoder decoder cross attention"
            enc = enc.to(self.device)
            cross_qkv = self.qkv(enc)
            # use keys and values from encoder output
            k = cross_qkv[:, :, self.head_dim:2*self.head_dim]
            v = cross_qkv[:, :, 2*self.head_dim:]
        z = self.dot_product(q, k, v, mask)
        return z

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot product for self attention
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, device='cpu'):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        # softmax along last dimension
        self.softmax = nn.Softmax(dim=-1).to(device)

    def forward(self, q, k, v, mask=None):
        """
        scaled dot product forward method
        :param q: type tensor, query embeddings
        :param k: type tensor, key embeddings
        :param v: type tensor, value embeddings
        :param mask: attention mask
        :return:
            z: attn score weighted sum of values
        """
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        # q,k,v are 3d tensors: batch, length, embedding dim
        # embedding dimension = d_model//attn_heads
        _, seq_len, dk = k.shape
        # transpose along last two dimensions
        k_t = k.transpose(-2,-1)
        # scaled dot product
        scale = 1/torch.tensor([dk], device=self.device)
        attn_score = (q @ k_t) * scale
        # apply mask
        if mask is not None:
            mask = mask.to(self.device)
            assert mask.dim() == 2 and mask.shape == (seq_len, seq_len), f"mask should be a 2D 'seq_length x seq_length' tensor."
            # fill maksed positions with low value so it becomes 0 in softmax
            attn_score = attn_score.masked_fill(mask == 0, -1000)
        attn_score = self.softmax(attn_score)
        # weigh values as per attention scores
        z = attn_score @ v
        return z

# test
if __name__ == '__main__':
    d_model = 128
    num_attn_heads = 8
    batch_size = 2
    seq_len = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_ids = torch.randint(0, 10, size=(batch_size, seq_len)).to(device)
    embedding = InputEmbedding(vocab_size=20, max_seq_len=10, d_model=d_model, device=device)
    token_emb = embedding(token_ids)
    multi_head = MultiHeadSelfAttention(d_model, num_attn_heads, device=device)
    # causal mask
    mask = torch.ones(seq_len, seq_len).to(device)
    mask = torch.tril(mask)
    z = multi_head(token_emb, mask)
    layer_norm = LayerNorm(d_model, device=device)
    z = layer_norm(z)
    ff = FeedForward(d_model, 64, device=device)
    z = ff(z)
    print(f"out shape: {z.shape}")