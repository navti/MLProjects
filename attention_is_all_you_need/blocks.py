import torch
from torch import nn
from layers import *


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_attn_heads):
        super(EncoderBlock, self).__init__()
        self.multi_head_attn = MultiHeadSelfAttention(d_model, num_attn_heads)
        self.layer_norm = LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        multihead_out = self.multi_head_attn(x)
        addnorm_out1 = self.layer_norm(x + multihead_out)
        ff_out = self.ff(addnorm_out1)
        addnorm_out2 = self.layer_norm(addnorm_out1 + ff_out)
        return addnorm_out2

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_attn_heads):
        super(DecoderBlock, self).__init__()
        self.multi_head_attn = MultiHeadSelfAttention(d_model, num_attn_heads)
        # self.multi_head_attn2 = MultiHeadSelfAttention(d_model, num_attn_heads)
        self.ff = FeedForward(d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, enc=None, mask=None):
        # multihead attn and add and norm 1
        multihead_out1 = self.multi_head_attn(x, mask)
        addnorm_out1 = self.layer_norm(x + multihead_out1)
        # multi head cross attn and addnorm layer 2
        multihead_out2 = self.multi_head_attn(addnorm_out1, mask, True, enc)
        addnorm_out2 = self.layer_norm(addnorm_out1 + multihead_out2)
        # feed forward, add norm layer last
        ff_out = self.ff(addnorm_out2)
        addnorm_out3 = self.layer_norm(addnorm_out2 + ff_out)
        return addnorm_out3

# test
if __name__ == '__main__':
    d_model = 128
    num_attn_heads = 8
    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 10, size=(batch_size, seq_len))
    embedding = InputEmbedding(vocab_size=20, max_seq_len=10, d_model=d_model)
    token_emb = embedding(token_ids)
    encoder = EncoderBlock(d_model, num_attn_heads)
    enc_out = encoder(token_emb)
    decoder = DecoderBlock(d_model, num_attn_heads)
    dec_out = decoder(token_emb, enc_out)
    print(f"out shape: {dec_out.shape}")