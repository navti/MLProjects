import torch
import argparse
from bpe.tokenizer import load_tokenizer, BPETokenizer
from model import Translator
from dataset import TranslateSet
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

def get_args():
    parser = argparse.ArgumentParser(description='Train English to Hindi translator')
    parser.add_argument('--dmodel', '-d', metavar='M', type=int, default=128, help='Model dimension')
    parser.add_argument('--attn-heads', '-a', metavar='A', type=int, default=8, help='No. of self attention heads')
    parser.add_argument('--nenc', metavar='E', type=int, default=6, help='No. of encoder blocks in the transformer')
    parser.add_argument('--ndec', metavar='D', type=int, default=6, help='No. of decoder blocks in the transformer')
    parser.add_argument('--context-length', '-c', metavar='C', type=int, default=32, help='Max no. of tokens in the input sequence')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='Train model with given parameters.')
    parser.add_argument('--save', '-n', type=str, default="model.pth", help='Save model with name.')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

if __name__ == "__main__":
    data_dir = "./data"
    train_csv = f"{data_dir}/train.csv"
    test_csv = f"{data_dir}/test.csv"
    validation_csv = f"{data_dir}/validation.csv"
    tokenizer_dir = './bpe/saved'
    # en_tokenizer = BPETokenizer()
    # hi_tokenizer = BPETokenizer()
    en_tokenizer = load_tokenizer(f"{tokenizer_dir}/en_tokenizer.pkl")
    hi_tokenizer = load_tokenizer(f"{tokenizer_dir}/hi_tokenizer.pkl")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    args = get_args()
    args.train = True
    model_kwargs = {}
    model_kwargs['num_enc'] = args.nenc
    model_kwargs['num_dec'] = args.ndec
    model_kwargs['d_model'] = args.dmodel
    model_kwargs['num_attn_heads'] = args.attn_heads
    model_kwargs['enc_vocab_size'] = len(en_tokenizer.vocab)
    model_kwargs['dec_vocab_size'] = len(hi_tokenizer.vocab)
    model_kwargs['max_seq_len'] = args.context_length
    model_kwargs['enc_pad_idx'] = en_tokenizer.special_tokens["[PAD]"]
    model_kwargs['dec_pad_idx'] = hi_tokenizer.special_tokens["[PAD]"]
    model_kwargs['device'] = device

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    validation_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)

    train_set = TranslateSet(train_csv, args.context_length, en_tokenizer, hi_tokenizer)
    test_set = TranslateSet(train_csv, args.context_length, en_tokenizer, hi_tokenizer)
    validation_set = TranslateSet(train_csv, args.context_length, en_tokenizer, hi_tokenizer)

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)
    validation_loader = DataLoader(validation_set, **validation_kwargs)

    model = Translator(**model_kwargs).to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-2, lr=args.lr, betas=(0.9, 0.999))

    # training loop
    # for epoch in range(1, args.epochs+1):

    enc_in, dec_in, pad_mask = next(iter(train_loader))
    enc_in = enc_in.to(device)
    dec_in = dec_in.to(device)
    pad_mask = pad_mask[:,1:].to(device)
    targets = dec_in[:,1:]
    targets = (targets * pad_mask).long()
    optimizer.zero_grad()
    predictions = model(enc_in, dec_in[:,:-1])
    batch_size, context_len, d = predictions.shape
    pad_mask = pad_mask.unsqueeze(dim=-1).expand(batch_size, context_len, d).clone()
    predictions = predictions * pad_mask
    pad_mask[:,:,0] = 1 - pad_mask[:,:,0]
    pad_mask[:,:,1:] = 0
    predictions += pad_mask
    predictions = torch.einsum('pqr->prq', predictions)
    batch_loss = loss_criterion(predictions, targets)

    
    print(targets.shape)