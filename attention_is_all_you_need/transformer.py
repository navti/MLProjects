import torch
from torch import nn
from blocks import *
from torchinfo import summary

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_attn_heads, vocab_size, max_seq_len, pad_idx=None):
        super(TransformerEncoder, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model, pad_idx)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, num_attn_heads) for _ in range(num_blocks)])

    def forward(self, token_ids, mask=None):
        token_embeddings = self.input_embedding(token_ids)
        out = self.enc_blocks[0](token_embeddings, mask)
        for enc_block in self.enc_blocks[1:]:
            out = enc_block(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_attn_heads, vocab_size, max_seq_len, pad_idx=None):
        super(TransformerDecoder, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model, pad_idx)
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, num_attn_heads) for _ in range(num_blocks)])

    def forward(self, token_ids, enc_out, mask=None):
        token_embeddings = self.input_embedding(token_ids)
        out = self.dec_blocks[0](token_embeddings, enc_out, mask)
        for dec_block in self.dec_blocks[1:]:
            out = dec_block(out, enc_out)
        return out

class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__()
        self.num_enc = kwargs.get('num_enc') if 'num_enc' in kwargs else args[0]
        self.num_dec = kwargs.get('num_dec') if 'num_dec' in kwargs else args[1]
        self.d_model = kwargs.get('d_model') if 'd_model' in kwargs else args[2]
        self.num_attn_heads = kwargs.get('num_attn_heads') if 'num_attn_heads' in kwargs else args[3]
        self.vocab_size = kwargs.get('vocab_size') if 'vocab_size' in kwargs else args[4]
        self.max_seq_len = kwargs.get('max_seq_len') if 'max_seq_len' in kwargs else args[5]
        self.pad_idx = kwargs.get('pad_idx') if 'pad_idx' in kwargs else args[6] if len(args) == 7 else None

        self.encoder = TransformerEncoder(self.num_enc, self.d_model, self.num_attn_heads, self.vocab_size, self.max_seq_len, self.pad_idx)
        self.decoder = TransformerDecoder(self.num_dec, self.d_model, self.num_attn_heads, self.vocab_size, self.max_seq_len, self.pad_idx)

    def forward(self, enc_token_ids, dec_token_ids, enc_mask=None, dec_mask=None):
        enc_out = self.encoder(enc_token_ids, enc_mask)
        dec_out = self.decoder(dec_token_ids, enc_out, dec_mask)
        return dec_out

# test
if __name__ == "__main__":
    num_enc = 3
    num_dec = 5
    d_model = 128
    num_attn_heads = 8
    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 10, size=(batch_size, seq_len))
    mask = torch.tensor([1,1,1,0,0])
    tr = Transformer(num_enc, num_dec, d_model, num_attn_heads, vocab_size=20, max_seq_len=10)
    out = tr(token_ids, token_ids, None, mask)
    summary(tr)
    print(out.shape)