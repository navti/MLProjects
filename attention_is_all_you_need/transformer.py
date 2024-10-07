import torch
from torch import nn
from blocks import *
from torchinfo import summary

class TransformerEncoder(nn.Module):
    """
    Transformer encoder class containing encoder blocks of the transformer
    :param num_blocks: type int, no. of encoder blocks in transformer
    :param d_model: type int, embedding dimension of the model
    :param num_attn_heads: type int, no. of self attention heads inside the multi head layer
    :param vocab_size: type int, no. of rows in the embedding table, size of vocabulary.
    :param max_seq_len: type int, max sequence length allowed, used by positional encoding layer
    :param pad_idx: type int, the padding token id
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, num_blocks, d_model, num_attn_heads, vocab_size, max_seq_len, pad_idx=None, device='cpu'):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model, pad_idx, device=device)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, num_attn_heads, device=device) for _ in range(num_blocks)])

    def forward(self, token_ids, mask=None):
        """
        forward method of transformer encoder
        :param token_ids: type tensor[int], size: batch, max_seq_len, input token ids padded to max_seq_len
        :param mask: type tensor[int], size: max_seq_len, attention mask
        :return:
            out: type tensor, size: batch, max_seq_len, d_model
        """
        token_ids = token_ids.to(self.device)
        token_embeddings = self.input_embedding(token_ids)
        out = self.enc_blocks[0](token_embeddings, mask)
        for enc_block in self.enc_blocks[1:]:
            out = enc_block(out)
        return out

class TransformerDecoder(nn.Module):
    """
    Transformer decoder class containing decoder blocks of the transformer
    :param num_blocks: type int, no. of encoder blocks in transformer
    :param d_model: type int, embedding dimension of the model
    :param num_attn_heads: type int, no. of self attention heads inside the multi head layer
    :param vocab_size: type int, no. of rows in the embedding table, size of vocabulary.
    :param max_seq_len: type int, max sequence length allowed, used by positional encoding layer
    :param pad_idx: type int, the padding token id
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, num_blocks, d_model, num_attn_heads, vocab_size, max_seq_len, pad_idx=None, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model, pad_idx, device=device)
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, num_attn_heads, device=device) for _ in range(num_blocks)])

    def forward(self, token_ids, enc_out, mask=None):
        """
        forward method of transformer encoder
        :param token_ids: type tensor[int], size: batch, max_seq_len, input token ids padded to max_seq_len
        :param enc_out: type tensor, size: batch, max_seq_len, d_model
        :param mask: type tensor[int], size: max_seq_len, attention mask
        :return:
            out: type tensor, size: batch, max_seq_len, d_model
        """
        token_ids = token_ids.to(self.device)
        token_embeddings = self.input_embedding(token_ids)
        out = self.dec_blocks[0](token_embeddings, enc_out, mask)
        for dec_block in self.dec_blocks[1:]:
            out = dec_block(out, enc_out)
        return out

class Transformer(nn.Module):
    """
    Transformer encoder class containing encoder blocks of the transformer
    :param num_enc: type int, no. of encoder blocks in transformer
    :param num_dec: type int, no. of decoder blocks in transformer
    :param d_model: type int, embedding dimension of the model
    :param num_attn_heads: type int, no. of self attention heads inside the multi head layer
    :param vocab_size: type int, no. of rows in the embedding table, size of vocabulary.
    :param max_seq_len: type int, max sequence length allowed, used by positional encoding layer
    :param pad_idx: type int, the padding token id
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__()
        self.num_enc = kwargs.get('num_enc') if 'num_enc' in kwargs else args[0]
        self.num_dec = kwargs.get('num_dec') if 'num_dec' in kwargs else args[1]
        self.d_model = kwargs.get('d_model') if 'd_model' in kwargs else args[2]
        self.num_attn_heads = kwargs.get('num_attn_heads') if 'num_attn_heads' in kwargs else args[3]
        self.vocab_size = kwargs.get('vocab_size') if 'vocab_size' in kwargs else args[4]
        self.max_seq_len = kwargs.get('max_seq_len') if 'max_seq_len' in kwargs else args[5]
        self.pad_idx = kwargs.get('pad_idx') if 'pad_idx' in kwargs else args[6] if len(args) >= 7 else None
        self.device = kwargs.get('device') if 'device' in kwargs else args[7] if len(args) == 8 else 'cpu'

        self.encoder = TransformerEncoder(self.num_enc, self.d_model, self.num_attn_heads, self.vocab_size, self.max_seq_len, self.pad_idx, device=self.device)
        self.decoder = TransformerDecoder(self.num_dec, self.d_model, self.num_attn_heads, self.vocab_size, self.max_seq_len, self.pad_idx, device=self.device)

    def forward(self, enc_token_ids, dec_token_ids, enc_mask=None, dec_mask=None):
        """
        forward method of transformer encoder
        :param enc_token_ids: type tensor[int], size: batch, max_seq_len, encoder input token ids padded to max_seq_len
        :param dec_token_ids: type tensor[int], size: batch, max_seq_len, decoder input token ids padded to max_seq_len
        :param enc_mask: type tensor[int], size: max_seq_len, attention mask for encoder
        :param dec_mask: type tensor[int], size: max_seq_len, attention mask for decoder
        :return:
            dec_out: type tensor, size: batch, max_seq_len, d_model
        """
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_ids = torch.randint(0, 10, size=(batch_size, seq_len)).to(device)
    mask = torch.tensor([1,1,1,0,0]).to(device)
    tr = Transformer(num_enc, num_dec, d_model, num_attn_heads, vocab_size=20, max_seq_len=10, device=device)
    out = tr(token_ids, token_ids, None, mask)
    summary(tr)
    print(out.shape)