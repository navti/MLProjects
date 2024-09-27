import torch
from torch import nn
from embeddings import InputEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attn_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.self_attn_heads = [SelfAttentionHead(d_model, num_attn_heads) for _ in range(num_attn_heads)]

    def forward(self, x):
        # x token embeddings, batch_size x length x d_model
        out = []
        for self_attn_head in self.self_attn_heads:
            out.append(self_attn_head(x))
        z = torch.cat(out, dim=-1)
        return self.linear(z)

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, num_attn_heads):
        super(SelfAttentionHead, self).__init__()
        self.head_dim = d_model//num_attn_heads
        assert self.head_dim * num_attn_heads == d_model, "model dimension not a multiple of attn head dimension."
        self.qkv = nn.Linear(d_model, 3*self.head_dim, bias=False)
        self.dot_product = ScaledDotProductAttention()
    
    def forward(self, x):
        # x token embeddings, batch_size x length x d_model
        qkv = self.qkv(x)
        q = qkv[:, :, :self.head_dim]
        k = qkv[:, :, self.head_dim:2*self.head_dim]
        v = qkv[:, :, 2*self.head_dim:]
        z = self.dot_product(q, k, v)
        return z

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        # softmax along last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q,k,v are 3d tensors: batch, length, embedding dim
        # embedding dimension = d_model//attn_heads
        _, _, dk = k.shape
        # transpose along last two dimensions
        k_t = k.transpose(-2,-1)
        # scaled dot product
        scale = 1/torch.tensor([dk])
        attn_score = (q @ k_t) * scale
        # apply mask
        if mask is not None:
            # fill maksed positions with low value so it becomes 0 in softmax
            attn_score = attn_score.masked_fill(mask == 0, -1000)
        attn_score = self.softmax(attn_score)
        # weigh values as per attention scores
        z = attn_score @ v
        return z
    
if __name__ == '__main__':
    token_ids = torch.randint(0, 10, size=(2,5))
    embedding = InputEmbedding(20, 10, d_model=128)
    token_emb = embedding(token_ids)
    multi_head = MultiHeadSelfAttention(d_model=128, num_attn_heads=8)
    z = multi_head(token_emb)
    print("end")