import torch
from torch import optim
from torch import nn
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from model import *
import argparse
import glob
from collections import defaultdict
import sys
# add to path variable so module can be found
root_dir = '/'.join(__file__.split('/')[:-2])
sys.path.append(root_dir)
torch.set_float32_matmul_precision('high')
from utils.cifar_utils import *

def print_training_parameters(args, device):
    print(f"")
    print(f"=========== Training Parameters ===========")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"No. of filters: {args.nf}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Baseline beta: {args.baseline_beta}")
    print(f"Annealing steps: {args.annealing_steps}")
    print(f"Annealing shape: {args.annealing_shape}")
    print(f"Annealing disabled: {args.annealing_disable}")
    print(f"Cyclic annealing disabled: {args.annealing_cyclic_disable}")
    print(f"===========================================")
    print(f"")


def train_model(model, device, train_loader, optimizer, loss_criterion, epoch):
    model.train()
    total_gen_loss = 0
    total_d_loss = 0
    for batch_idx, (real_data, _) in enumerate(train_loader):
        batch_size = len(real_data)
        # real samples
        real_data = real_data.to(device)
        ones = torch.ones((batch_size), dtype=torch.float32).to(device)
        # sample from normal distribution
        latent_vectors = torch.randn(size=(batch_size, model.latent_dim), dtype=torch.float32).to(device)
        zeros = torch.zeros((batch_size), dtype=torch.float32).to(device)
        optimizer.zero_grad()
        # discriminator loss on real data
        real_preds = model.discriminator(real_data)
        real_loss = loss_criterion(real_preds, ones)
        # get generator output and discriminator preds on it
        _, fake_preds = model(latent_vectors)
        # discriminator loss on fake images
        fake_loss = loss_criterion(fake_preds, zeros)
        # total discriminator loss
        d_loss = real_loss + fake_loss
        # compute all gradients
        d_loss.backward()
        total_d_loss += d_loss.item()
        # zero out gradients for generator computed in last step
        # gradients for generator will be different
        model.generator.zero_grad()
        # update discriminator weights, generator grads are zero, so no update will happen there
        optimizer.step()
        optimizer.zero_grad()
        # compute loss for discriminator whose gradients will be used to update the generator
        g_loss = loss_criterion(fake_preds, ones)
        g_loss.backward()
        total_gen_loss += g_loss.item()
        # zero out discriminator grad
        model.discriminator.zero_grad()
        optimizer.step()
        # gather losses for logging

    avg_gen_loss = total_gen_loss/(batch_idx+1)
    avg_d_loss = total_d_loss/(batch_idx+1)
    avg_total_loss = avg_gen_loss + avg_d_loss
    print(f"Train epoch: {epoch} [{(batch_idx+1) * len(real_data)}/{len(train_loader.dataset)}]\
          \tGen Loss: {avg_gen_loss:.6f}\tDiscriminator Loss: {avg_d_loss:.6f}")
    return avg_gen_loss, avg_d_loss, avg_total_loss


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 GAN")
    parser.add_argument('--input-channels', type=int, default=3, metavar='IN',
                        help='input channels in the images. (default: 3)')
    parser.add_argument('--n-classes', type=int, default=10, metavar='C',
                        help='no. of classes. (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--nf', type=int, default=32, metavar='NF',
                        help='no. of filters (default: 32)')
    parser.add_argument('--latent-dim', type=int, default=100, metavar='LD',
                        help='size of latent dimension (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'using device: {device}')
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data_dir = root_dir+"/data/"
    train_files = glob.glob(data_dir + '**/*_batch_*', recursive=True)
    train_files = list(map(unpickle, train_files))
    train = combine_dicts(train_files)
    print(f"train data shape: {train['data'].shape}")

    # make train and test dataset
    trainset = make_dataset(train)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)

    model = GAN(args.latent_dim).to(device)
    opt_model = torch.compile(model)
    optimizer = optim.Adadelta(opt_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    loss_criterion = nn.BCELoss()
    #print_model_summary(model)

    print_training_parameters(args, device)

    # Run training loops
    losses = defaultdict(list)
    for epoch in range(1,args.epochs+1):
        avg_loss, avg_recon_loss, avg_kl_loss, avg_cl_loss = train_model(opt_model, device, train_loader, optimizer, loss_criterion, epoch)
        losses['total_loss'].append(avg_loss)
        losses['recon_loss'].append(avg_recon_loss)
        losses['kl_loss'].append(avg_kl_loss)
        losses['cl_loss'].append(avg_cl_loss)
        scheduler.step()
        if (args.dry_run):
            break

    results_dir = root_dir+"/VAE/results"
    models_dir = results_dir+"/saved_models"
    inference_dir = results_dir+"/generated"

    save_plots(losses, results_dir)
    # save model
    model_name = f"VAE_CIFAR10_lr_{args.lr}"
    if (args.save_model):
        save_model(model, models_dir, name=model_name)

    # generate samples using the model
    generate_samples(5,10,opt_model, inference_dir)

if __name__ == '__main__':
    main()