import torch
import argparse
from bpe.tokenizer import load_tokenizer, BPETokenizer
from model.model import *
from data.dataset import TranslateSet
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import sys

def get_args():
    parser = argparse.ArgumentParser(description='English to Hindi translator')
    parser.add_argument('--dmodel', '-d', metavar='MODEL_DIM', type=int, default=128, help='Model dimension')
    parser.add_argument('--attn-heads', '-a', metavar='NUM_HEADS', type=int, default=8, help='No. of self attention heads')
    parser.add_argument('--nenc', metavar='NUM_ENC', type=int, default=6, help='No. of encoder blocks in the transformer')
    parser.add_argument('--ndec', metavar='NUM_DEC', type=int, default=6, help='No. of decoder blocks in the transformer')
    parser.add_argument('--context-length', '-c', metavar='LENGTH', type=int, default=32, help='Max no. of tokens in the input sequence')
    parser.add_argument('--epochs', '-e', metavar='NUM_EPOCHS', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='BATCH_SIZE', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='Train model with given parameters.')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--load', '-f', metavar='MODEL_PATH', type=str, default="", help='Load model from a .pth file. Required when running inference.')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser, parser.parse_args()

if __name__ == "__main__":
    proj_dir = '/'.join(__file__.split('/')[:-1])
    data_dir = f"{proj_dir}/data"
    train_csv = f"{data_dir}/train.csv"
    test_csv = f"{data_dir}/test.csv"
    validation_csv = f"{data_dir}/validation.csv"
    tokenizer_dir = f"{proj_dir}/bpe/saved"
    en_tokenizer = load_tokenizer(f"{tokenizer_dir}/en_tokenizer.pkl")
    hi_tokenizer = load_tokenizer(f"{tokenizer_dir}/hi_tokenizer.pkl")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    parser, args = get_args()
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

    if args.train:
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
        test_set = TranslateSet(test_csv, args.context_length, en_tokenizer, hi_tokenizer)
        validation_set = TranslateSet(validation_csv, args.context_length, en_tokenizer, hi_tokenizer)

        train_loader = DataLoader(train_set, **train_kwargs)
        test_loader = DataLoader(test_set, **test_kwargs)
        validation_loader = DataLoader(validation_set, **validation_kwargs)

        model = Translator(**model_kwargs).to(device)
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), weight_decay=1e-2, lr=args.lr, betas=(0.9, 0.999))

        # training loop
        losses = {'train':[], 'validation':[]}
        scores = {'bleu':[]}
        for epoch in range(1, args.epochs+1):
            # train
            avg_epoch_loss = train(model, device, epoch, optimizer, train_loader, loss_criterion)
            losses['train'].append(avg_epoch_loss)
            # evaluate
            avg_loss, avg_bleu_score = evaluate(model, device, epoch, validation_loader, loss_criterion, hi_tokenizer)
            losses['validation'].append(avg_loss)
            scores['bleu'].append(avg_bleu_score)

        results_dir = f"{proj_dir}/results"
        models_dir = f"{proj_dir}/model/saved_models"
        save_plots(losses, results_dir, name="losses")
        save_plots(scores, results_dir, name="scores")
        # save model
        model_name = f"translateEnHi"
        if (args.save_model):
            save_model(model, models_dir, name=model_name)
    else:
        models_dir = f"{proj_dir}/model/saved_models"
        model = load_model(args.load, **model_kwargs)
        if model == None:
            print("Check if the correct model path was provided to load from.")
            parser.print_help(sys.stderr)
            sys.exit(1)
        # run inference loop
        model = model.to(device)
        while True:
            en_sent = input("Enter an English sentence.\n")
            hi_sent = infer(en_sent, args.context_length, model, device, en_tokenizer, hi_tokenizer)
            print(hi_sent)