import torch
from torch import nn
from base.layers import *

class EncoderBlock(nn.Module):
    """
    Encoder block
    :param d_model: type int, token embedding dimension
    :param num_attn_heads: type int, no. of attention heads
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, num_attn_heads, device='cpu'):
        super(EncoderBlock, self).__init__()
        self.device = device
        self.multi_head_attn = MultiHeadSelfAttention(d_model, num_attn_heads, device)
        self.layer_norm1 = LayerNorm(d_model, device)
        self.layer_norm2 = LayerNorm(d_model, device)
        self.ff = FeedForward(d_model, device=device)

    def forward(self, x, mask=None):
        """
        call encoder block
        :param x: type tensor, token embeddings
        :param mask: attention mask
        """
        x = x.to(self.device)
        multihead_out = self.multi_head_attn(x, mask)
        addnorm_out1 = self.layer_norm1(x + multihead_out)
        ff_out = self.ff(addnorm_out1)
        addnorm_out2 = self.layer_norm2(addnorm_out1 + ff_out)
        return addnorm_out2

class DecoderBlock(nn.Module):
    """
    Decoder block
    :param d_model: type int, token embedding dimension
    :param num_attn_heads: type int, no. of attention heads
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, d_model, num_attn_heads, device='cpu'):
        super(DecoderBlock, self).__init__()
        self.device = device
        self.multi_head_attn1 = MultiHeadSelfAttention(d_model, num_attn_heads, device)
        self.multi_head_attn2 = MultiHeadSelfAttention(d_model, num_attn_heads, device)
        self.layer_norm1 = LayerNorm(d_model, device)
        self.layer_norm2 = LayerNorm(d_model, device)
        self.layer_norm3 = LayerNorm(d_model, device)
        self.ff = FeedForward(d_model, device=device)

    def forward(self, x, enc=None, mask=None):
        """
        call decoder block
        :param x: type tensor, token embeddings
        :param enc: type tensor, encoder output to use for cross attention
        :param mask: attention mask
        """
        x = x.to(self.device)
        # multihead attn and add and norm 1
        multihead_out1 = self.multi_head_attn1(x, mask)
        addnorm_out1 = self.layer_norm1(x + multihead_out1)
        # multi head cross attn and addnorm layer 2
        multihead_out2 = self.multi_head_attn2(addnorm_out1, mask, True, enc)
        addnorm_out2 = self.layer_norm2(addnorm_out1 + multihead_out2)
        # feed forward, add norm layer last
        ff_out = self.ff(addnorm_out2)
        addnorm_out3 = self.layer_norm3(addnorm_out2 + ff_out)
        return addnorm_out3

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
    encoder = EncoderBlock(d_model, num_attn_heads, device=device)
    enc_out = encoder(token_emb)
    decoder = DecoderBlock(d_model, num_attn_heads, device=device)
    dec_out = decoder(token_emb, enc_out)
    print(f"out shape: {dec_out.shape}")