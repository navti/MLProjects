import torch
from torch import nn
from base.transformer import Transformer
from torch.optim import optimizer
from utils import *
import pathlib
import matplotlib.pyplot as plt
import time

class Translator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Translator, self).__init__()
        self.transformer = Transformer(*args, **kwargs)
        self.linear = nn.Linear(self.transformer.d_model, self.transformer.dec_vocab_size)
        self.dec_mask = torch.tril(torch.ones(self.transformer.max_seq_len, self.transformer.max_seq_len))

    def forward(self, in_token_ids, target_token_ids):
        tr_out = self.transformer(in_token_ids, target_token_ids, None, self.dec_mask)
        logits = self.linear(tr_out)
        return logits

def adjust_predictions(predictions, p_mask, pad_id):
    batch_size, context_len, vocab_size = predictions.shape
    pad_mask = p_mask.clone()
    pad_mask = pad_mask.unsqueeze(dim=-1).expand(batch_size, context_len, vocab_size).clone()
    predictions = predictions * pad_mask.clone()
    pad_mask[:,:,pad_id] = 1 - pad_mask[:,:,pad_id]
    pad_mask[:,:,:pad_id] = 0
    pad_mask[:,:,pad_id+1:] = 0
    predictions = predictions + pad_mask.clone()
    return predictions

def infer(sent, context_len, model, device, en_tokenizer, hi_tokenizer):
    model.eval()
    enc_in, _ = process_sentence(sent, context_len, en_tokenizer)
    enc_in = enc_in.unsqueeze(dim=0).to(device)
    out_tokens = []
    sos = hi_tokenizer.special_tokens["[SOS]"]
    eos = hi_tokenizer.special_tokens["[EOS]"]
    pad = hi_tokenizer.special_tokens["[PAD]"]
    next_token = sos
    while next_token != eos and len(out_tokens) < context_len:
        out_tokens.append(next_token)
        dec_in = out_tokens + [pad for _ in range(context_len - len(out_tokens))]
        dec_in = torch.tensor(dec_in).unsqueeze(dim=0).to(device)
        dec_out = model(enc_in, dec_in)
        predictions = dec_out.argmax(dim=-1).squeeze()
        next_token = predictions[len(out_tokens)-1].item()
    return hi_tokenizer.decode_nice(out_tokens)

def train(model, device, epoch, optimizer, train_loader, loss_criterion, grad_scaler):
    model.train()
    dec_pad_idx = model.transformer.dec_pad_idx
    total_loss = 0
    for batch_idx, (enc_in, dec_in, pad_mask) in enumerate(train_loader):
        batch_size = enc_in.shape[0]
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        pad_mask = pad_mask[:,1:].to(device)
        targets = dec_in[:,1:]
        with torch.autocast(device.type, enabled=True):
            predictions = model(enc_in, dec_in[:,:-1])
            predictions = adjust_predictions(predictions, pad_mask, dec_pad_idx)
            predictions = torch.einsum('pqr->prq', predictions)
            loss = loss_criterion(predictions, targets)
            total_loss += (loss.item() / batch_size)
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if batch_idx % 100 == 0:
            print(f"Avg loss after batch {batch_idx+1}/{len(train_loader)}: {(total_loss/(batch_idx+1)):.4f}")
    avg_epoch_loss = total_loss/(batch_idx+1)
    print(f"Epoch {epoch}: Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def evaluate(model, device, epoch, validation_loader, loss_criterion, tokenizer):
    model.eval()
    dec_pad_idx = model.transformer.dec_pad_idx
    total_loss = 0
    total_bleu_score = 0
    for batch_idx, (enc_in, dec_in, pad_mask) in enumerate(validation_loader):
        batch_size = enc_in.shape[0]
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        pad_mask = pad_mask[:,1:].to(device)
        targets = dec_in[:,1:]
        predictions = model(enc_in, dec_in[:,:-1])
        predictions = adjust_predictions(predictions, pad_mask, dec_pad_idx)
        out_tokens = predictions.argmax(dim=-1).squeeze()
        predictions = torch.einsum('pqr->prq', predictions)
        loss = loss_criterion(predictions, targets)
        total_loss += (loss.item() / batch_size)
        total_bleu_score += get_batch_bleu_score(out_tokens, targets, tokenizer)
    avg_validation_loss = total_loss/(batch_idx+1)
    avg_bleu_score = total_bleu_score/(batch_idx+1)
    print(f"Validation after Epoch {epoch}:\n\tLoss: {avg_validation_loss:.4f}\n\tAvg BLEU score: {avg_bleu_score:.4f}")
    return avg_validation_loss, avg_bleu_score

def save_model(model, models_dir, name=None):
    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_model_path = models_dir+"/translate-"+timestr+".pth"
    else:
        save_model_path = models_dir+"/"+name+".pth"
    try:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved at: {save_model_path}")
    except OSError as e:
        print("Failed to save model.")
        print(f"{e.strerror}: {e.filename}")

# load model
def load_model(model_path, *args, **kwargs):
    model = Translator(*args, **kwargs)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except OSError as e:
        print(f"{e.strerror}: {e.filename}")
        return None
    return model

def save_plots(losses, results_dir, name=None):
    fig = plt.figure()
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = results_dir+"/plot-"+timestr
    else:
        save_plot_path = results_dir+"/"+name
    epochs = len(losses['train'] if 'train' in losses else losses['bleu'])
    if 'train' in losses:
        plt.plot(range(1,epochs+1), losses['train'], label='Train loss')
        plt.plot(range(1,epochs+1), losses['validation'], label='Validation loss')
        plt.ylabel('Loss')
    else:
        plt.plot(range(1,epochs+1), losses['bleu'], label='BLEU score')
        plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title(f"Translate EN HI")
    plt.savefig(save_plot_path, facecolor='w', edgecolor='none')
    print(f"Plot saved at: {save_plot_path}")