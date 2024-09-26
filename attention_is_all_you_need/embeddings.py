import torch
from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, pad_idx=None):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=pad_idx)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for tokens, dimension same as d_model so it can be added to token embeddings
    """
    def __init__(self, max_seq_len, d_model, device='cpu'):
        super(PositionalEncoding, self).__init__()
        
        # initialize encoding table with zeros
        self.pos_encoding = torch.zeros(size=(max_seq_len, d_model), dtype=torch.float32, device=device)
        # position vector
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # add a dimension so braodcasting would be possible
        pos = pos.unsqueeze(dim=1)
        # as per the paper, the adjacent encoding values (sinusoids) are off by 90 degrees
        # exponent : 2i
        exp = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)
        # sin terms at idx 2i
        stop = d_model//2 # to cover for odd d_model in cosine
        self.pos_encoding[:,0::2] = torch.sin(pos / (10000 ** (exp / d_model)))
        # cos terms at idx 2i+1
        self.pos_encoding[:,1::2] = torch.cos(pos / (10000 ** (exp[:stop] / d_model)))

    def forward(self, x):
        # x: inut token ids, size: batch_size x seq_len
        batch_size, seq_len = x.shape
        return self.pos_encoding[:seq_len]

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=128, pad_idx=None, device='cpu'):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.token_embedding = TokenEmbedding(self.vocab_size, self.d_model, self.pad_idx)
        self.pos_encoding = PositionalEncoding(self.max_seq_len, self.d_model, device=device)

    def forward(self, x):
        return self.token_embedding(x) + self.pos_encoding(x)

if __name__ == '__main__':
    token_ids = torch.randint(0, 10, size=(2,5))
    embedding = InputEmbedding(20, 10, d_model=5)
    print(embedding(token_ids))