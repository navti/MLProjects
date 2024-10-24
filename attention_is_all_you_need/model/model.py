import torch
from torch import nn
from base.transformer import Transformer
from torch.optim import optimizer
from utils import *
import pathlib
import matplotlib.pyplot as plt
import time

class Translator(nn.Module):
    """
    Translator class containing transformer block and linear head
    :param num_enc: type int, no. of encoder blocks in transformer
    :param num_dec: type int, no. of decoder blocks in transformer
    :param d_model: type int, embedding dimension of the model
    :param num_attn_heads: type int, no. of self attention heads inside the multi head layer
    :param enc_vocab_size: for encoder, type int, no. of rows in the embedding table, size of vocabulary.
    :param dec_vocab_size: for decoder, type int, no. of rows in the embedding table, size of vocabulary.
    :param max_seq_len: type int, max sequence length allowed, used by positional encoding layer
    :param enc_pad_idx: type int, the padding token id for encoder
    :param dec_pad_idx: type int, the padding token id for decoder
    :param device: device to be used (cpu/cuda)
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize tranformer, linear head and the decoder mask
        """
        super(Translator, self).__init__()
        self.transformer = Transformer(*args, **kwargs)
        self.linear = nn.Linear(self.transformer.d_model, self.transformer.dec_vocab_size)
        # causal attention mask
        self.dec_mask = torch.tril(torch.ones(self.transformer.max_seq_len, self.transformer.max_seq_len))

    def forward(self, in_token_ids, target_token_ids):
        """
        forward call on Translator object
        """
        tr_out = self.transformer(in_token_ids, target_token_ids, None, self.dec_mask)
        logits = self.linear(tr_out)
        return logits

def adjust_predictions(predictions, p_mask, pad_id):
    """
    Adjust model predictions to have highest probabilities values for pad tokens in corressponding
    places as that in target tensor (indicated by pad mask).
    :param predictions: tensor, output of the model
    :param p_mask: tensor, pad mask indicating pad token positions in the target tensor
    :param pad_id: token id for pad token, type int
    :return:
        predictions: modified predictions tensor with pad tokens having highest probability in
        the places where target has pad tokens. This is for correct loss calculation.
    """
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
    """
    Run inference on the given English sentence
    :param sent: string, English sentence
    :param context_len: the context length of the model
    :param model: translator model
    :param device: device on which model should be run (cpu or cuda)
    :param en_tokenizer: English tokenizer
    :param hi_tokenizer: Hindi tokenizer
    :return: type str, translated sentence in Hindi
    """
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
    """
    Train model
    :param model: model to train
    :param device: device on which model should be trained
    :param epoch: current epoch number
    :param optimizer: optimizer
    :param train_loader: loader to load training data from
    :param loss_criterion: loss function to be used
    :param grad_scaler: gradient scaler to be used with amp
    :return:
        avg_epoch_loss: avg loss for current epoch
    """
    model.train()
    dec_pad_idx = model.transformer.dec_pad_idx
    total_loss = 0
    for batch_idx, (enc_in, dec_in, pad_mask) in enumerate(train_loader):
        batch_size = enc_in.shape[0]
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        pad_mask = pad_mask[:,1:].to(device)
        targets = dec_in[:,1:]
        # use automatic mixed precision training
        with torch.autocast(device.type, enabled=True):
            predictions = model(enc_in, dec_in[:,:-1])
            predictions = adjust_predictions(predictions, pad_mask, dec_pad_idx)
            predictions = torch.einsum('pqr->prq', predictions)
            loss = loss_criterion(predictions, targets)
            total_loss += (loss.item() / batch_size)
        optimizer.zero_grad()
        # scale gradients when doing backward pass, avoid vanishing gradients
        grad_scaler.scale(loss).backward()
        # unscale gradients before applying
        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if batch_idx % 100 == 0:
            print(f"Avg loss after batch {batch_idx+1}/{len(train_loader)}: {(total_loss/(batch_idx+1)):.4f}")
    avg_epoch_loss = total_loss/(batch_idx+1)
    print(f"Epoch {epoch}: Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def evaluate(model, device, epoch, validation_loader, loss_criterion, tokenizer):
    """
    Evaluate model
    :param model: model to train
    :param device: device on which model should be trained
    :param epoch: current epoch number
    :param optimizer: optimizer
    :param validation_loader: loader to load training data from
    :param loss_criterion: loss function to be used
    :param tokenizer: tokenizer for decoding predictions
    :return:
        avg_validaton_loss: avg validation loss after training for current epoch
        avg_bleu_score: avg BLEU score on validation set after current epoch
    """
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
    """
    Save model to disk
    :param model: model to be saved
    :param models_dir: directory where the model should be saved
    :param name: model file name
    :return: None
    """
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
    """
    Load model from disk
    :param model_path: model file path
    :param args: args for model to initialize
    :param kwargs: keyword arguments for model to initialize
    :return:
        model: loaded model
    """
    model = Translator(*args, **kwargs)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except OSError as e:
        print(f"{e.strerror}: {e.filename}")
        return None
    return model

def save_plots(losses, results_dir, name=None):
    """
    Save loss and score plots
    :param losses: dict with losses or BLEU score
    :param results_dir: directory where plots will be saved
    :param name: plot file name
    """
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