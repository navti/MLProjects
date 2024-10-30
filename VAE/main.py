import torch
import argparse
import glob
from torch import optim
from torch.optim.lr_scheduler import StepLR
from models import *
from annealer import Annealer
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

def train_model(model, device, train_loader, optimizer, loss_criterion, epoch, annealing_agent):
    model.train()
    train_loss = 0
    total_kl_loss = 0
    total_recon_loss = 0
    total_cl_loss = 0
    for batchidx, (data, targets) in enumerate(train_loader):
        #print(f"batch id: {batchidx}, data shape: {data.shape}")
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        xhat, x, mu, logvar, out_targets = model(data)
        classification_loss = loss_criterion(out_targets, targets)
        total_cl_loss += classification_loss.item()
        loss, recon_loss, kl_loss, kll = elbo_loss(xhat, x, mu, logvar, annealing_agent)
        #print(f"loss: {loss.item()}")
        total_loss = loss + classification_loss
        train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss.backward()
        optimizer.step()
    annealing_agent.step()
    avg_kl_loss = total_kl_loss/(batchidx+1)
    avg_recon_loss = total_recon_loss/(batchidx+1)
    avg_loss = train_loss/(batchidx+1)
    avg_cl_loss = total_cl_loss/(batchidx+1)
    print(f"Train epoch: {epoch} [{(batchidx+1) * len(data)}/{len(train_loader.dataset)}\
          ({(100. * batchidx/len(train_loader)):.0f}%)]\tLoss: {avg_loss:.6f}\t")
    return avg_loss, avg_recon_loss, avg_kl_loss, avg_cl_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 VAE')
    parser.add_argument('--input-channels', type=int, default=3, metavar='IN',
                        help='input channels in the images. (default: 3)')
    parser.add_argument('--n-classes', type=int, default=10, metavar='C',
                        help='no. of classes. (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--baseline-beta', type=float, default=0.0, metavar='beta',
                        help='beta value to start with (default: 0.0)')
    parser.add_argument('--annealing-steps', type=int, default=10, metavar='steps',
                        help='total steps in annealing (default: 10)')
    parser.add_argument('--annealing-shape', type=str, default='logistic', metavar='shape',
                        help='Annealing shape: [linear, cosine, logistic] (default: logistic)')
    parser.add_argument('--annealing-disable', action='store_true', default=False,
                        help='disables annealing for KL div. loss')
    parser.add_argument('--annealing-cyclic-disable', action='store_true', default=False,
                        help='disables cyclic annealing')
    parser.add_argument('--nf', type=int, default=32, metavar='NF',
                        help='no. of filters (default: 32)')
    parser.add_argument('--latent-dim', type=int, default=128, metavar='LD',
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
    #test_files = glob.glob(data_dir + '**/*test_batch*', recursive=True)
    train_files = list(map(unpickle, train_files))
    #test_files = list(map(unpickle, test_files))
    #test = combine_dicts(test_files)
    train = combine_dicts(train_files)
    #print(f"test data shape: {test['data'].shape}")
    print(f"train data shape: {train['data'].shape}")

    # make train and test dataset
    trainset = make_dataset(train)
    #testset = make_dataset(test)
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    #test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    model = VAE(args.input_channels, args.nf, args.latent_dim, args.n_classes, device).to(device)
    opt_model = torch.compile(model)
    optimizer = optim.Adadelta(opt_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    loss_criterion = nn.CrossEntropyLoss()
    #print_model_summary(model)

    # Annealing params
    annealing_shape = args.annealing_shape
    annealing_steps = args.annealing_steps
    baseline_beta = args.baseline_beta
    disable_cyclical = args.annealing_cyclic_disable
    annealing_disable = args.annealing_disable
    annealing_agent = Annealer(annealing_steps, annealing_shape, baseline_beta, disable_cyclical, annealing_disable)

    print_training_parameters(args, device)

    # Run training loops
    losses = defaultdict(list)
    for epoch in range(1,args.epochs+1):
        avg_loss, avg_recon_loss, avg_kl_loss, avg_cl_loss = train_model(opt_model, device, train_loader, optimizer, loss_criterion, epoch, annealing_agent)
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