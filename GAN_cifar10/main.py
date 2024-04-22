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
    print(f"===========================================")
    print(f"")

def save_gan_plots(losses, results_dir, name=None):
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = results_dir+"/plot-"+timestr
    else:
        save_plot_path = results_dir+"/"+name
    epochs = len(losses['d_loss'])
    plt.plot(range(1,epochs+1), losses['total_loss'], label='Total loss')
    plt.plot(range(1,epochs+1), losses['gen_loss'], label='Generator loss')
    plt.plot(range(1,epochs+1), losses['d_loss'], label='Discriminator loss')
    #plt.plot(range(1,epochs+1), losses['cl_loss'], label='CL loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"GAN CIFAR10")
    plt.savefig(save_plot_path, facecolor='w', edgecolor='none')
    print(f"Plot saved at: {save_plot_path}")

def generate_images(rows, cols, model, inference_dir, device, name=None):
    model.eval()
    pathlib.Path(inference_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_fig_path = inference_dir+"/samples_"+timestr
    else:
        save_fig_path = inference_dir+"/"+name
    z = torch.randn(size=(rows*cols, model.latent_dim), dtype=torch.float32).to(device)
    xhat = model.generator(z)
    xhat = torch.einsum('nchw->nhwc',xhat)
    xhat = xhat.view(rows, cols, *xhat.shape[1:]).detach().cpu()
    #print(xhat.shape)
    fig, axs = plt.subplots(rows, cols, figsize=(cols,rows))
    for A, I in zip(axs,xhat):
        for ax, img in zip(A,I):
            ax.set_aspect('equal')
            ax.axis('off')
            ax.imshow(img)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_fig_path, facecolor='w', edgecolor='none')

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
        fake_data = model.generator(latent_vectors)
        # detach fake data from computation graph when feeding to discriminator
        # because generator doesn't need to be updated in this step
        fake_preds = model.discriminator(fake_data.detach())
        # fake_data, fake_preds = model(latent_vectors)
        # discriminator loss on fake images
        fake_loss = loss_criterion(fake_preds, zeros)
        # total discriminator loss
        d_loss = real_loss + fake_loss
        total_d_loss += d_loss.item()
        # compute all gradients
        d_loss.backward(retain_graph=True)
        optimizer.step()

        # compute loss for generator using discriminator's output but generator's expectation
        fake_preds = model.discriminator(fake_data)
        g_loss = loss_criterion(fake_preds, ones)
        g_loss.backward(retain_graph=False)
        total_gen_loss += g_loss.item()
        # zero out discriminator grad
        model.discriminator.zero_grad()
        optimizer.step()

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
    # test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        # test_kwargs.update(cuda_kwargs)

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
    # opt_model = torch.compile(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    loss_criterion = nn.BCELoss()
    print(summary(model))

    print_training_parameters(args, device)

    # Run training loops
    losses = defaultdict(list)
    for epoch in range(1,args.epochs+1):
        avg_gen_loss, avg_d_loss, avg_total_loss = train_model(model, device, train_loader, optimizer, loss_criterion, epoch)
        losses['gen_loss'].append(avg_gen_loss)
        losses['d_loss'].append(avg_d_loss)
        losses['total_loss'].append(avg_total_loss)
        scheduler.step()
        if (args.dry_run):
            break

    results_dir = root_dir+"/GAN_cifar10/results"
    models_dir = results_dir+"/saved_models"
    inference_dir = results_dir+"/generated"

    save_gan_plots(losses, results_dir, name="plot")
    # save model
    model_name = f"GAN_CIFAR10_lr_{args.lr}"
    if (args.save_model):
        save_model(model, models_dir, name=model_name)

    # generate samples using the model
    generate_images(5,10,model, inference_dir, device, name="sample")

if __name__ == '__main__':
    main()