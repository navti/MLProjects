import torch
from torch import nn
from base.transformer import Transformer
from torch.optim import optimizer
from utils import bleu_score

class Translator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Translator, self).__init__()
        self.transformer = Transformer(*args, **kwargs)
        self.linear = nn.Linear(self.transformer.d_model, self.transformer.dec_vocab_size)
        self.dec_mask = torch.tril(torch.ones(self.transformer.max_seq_len, self.transformer.max_seq_len))

    def forward(self, in_token_ids, target_token_ids):
        tr_out = self.transformer(in_token_ids, target_token_ids, None, self.dec_mask)
        predicted_tokens = self.linear(tr_out)
        return predicted_tokens


def adjust_predictions(predictions, pad_mask):
    batch_size, context_len, vocab_size = predictions.shape
    pad_mask = pad_mask.unsqueeze(dim=-1).expand(batch_size, context_len, vocab_size).clone()
    predictions = predictions * pad_mask
    pad_mask[:,:,0] = 1 - pad_mask[:,:,0]
    pad_mask[:,:,1:] = 0
    predictions += pad_mask
    predictions = torch.einsum('pqr->prq', predictions)
    return predictions

def train(model, device, epoch, optimizer, train_loader, loss_criterion):
    model.train()
    total_loss = 0
    for batch_idx, (enc_in, dec_in, pad_mask) in enumerate(train_loader):
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        pad_mask = pad_mask[:,1:].to(device)
        targets = dec_in[:,1:]
        targets = (targets * pad_mask).long()
        optimizer.zero_grad()
        predictions = model(enc_in, dec_in[:,:-1])
        predictions = adjust_predictions(predictions, pad_mask)
        loss = loss_criterion(predictions, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Avg loss after batch {batch_idx+1}: {(total_loss/(batch_idx+1)):.2f}")
    print(f"Epoch {epoch}: Loss: {(total_loss/(batch_idx+1)):.2f}")
