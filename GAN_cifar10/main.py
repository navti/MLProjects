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
    """
    print training parameters of the model
    :param args: args containing params
    :param device: device being used to host the model
    :returns: None
    """
    print(f"")
    print(f"=========== Training Parameters ===========")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Generator Learning rate: {args.gen_lr}")
    print(f"Discriminator Learning rate: {args.d_lr}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"===========================================")
    print(f"")

def save_gan_plots(losses, results_dir, name=None):
    """
    save loss plots for GAN
    :param losses: dict object containing generator and discriminator losses
    :param results_dir: directory to store the plots in
    :param name: name of saved plot
    :returns: None
    """
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

def generate_images(rows, cols, model, inference_dir, device, z=None, name=None):
    """
    generate sample images using the model
    :param rows: rows of images in the figure
    :param cols: columns of images in the figure
    :param inference_dir: directory where the sample images to store
    :param device: device where model should be run
    :param z: the noise vector to use with generator, if none, one will be created
    :param name: name of figure to save
    :returns: None
    """
    pathlib.Path(inference_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_fig_path = inference_dir+"/samples_"+timestr
    else:
        save_fig_path = inference_dir+"/"+name
    if z == None:
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
    plt.close(fig)

def train_model(model, device, train_loader, gen_optimizer, d_optimizer,
                loss_criterion, clf_loss_criterion, epoch):
    """
    run one training loop
    :param model: model to train
    :param device: device hosting the model
    :param train_loader: data loader with train images
    :param gen_optimizer: optimizer used for the generator
    :param d_optimizer: optimizer used for the discriminator
    :param loss_criterion: loss function to use
    :param clf_loss_criterion: loss function to use for classification loss
    :param epoch: epoch/iteration number
    :returns:
        avg_gen_loss: generator loss
        avg_d_loss: discriminator loss
        avg_total_loss: total loss
    """
    model.train()
    total_gen_loss = 0
    total_d_loss = 0
    for batch_idx, (real_data, real_labels) in enumerate(train_loader):
        batch_size = len(real_data)
        noise = torch.normal(mean=0, std=0.1, size=real_data.size()).to(device)
        # noise /= epoch
        # real samples
        real_data = real_data.to(device)
        # real_data += noise/epoch
        real_labels = real_labels.to(device)
        ones = torch.ones((batch_size), dtype=torch.float32).to(device)
        # sample from normal distribution
        latent_vectors = model.generator.sample_latent_vectors(n_samples=batch_size)
        zeros = torch.zeros((batch_size), dtype=torch.float32).to(device)

        d_optimizer.zero_grad()
        # discriminator loss on real data
        real_preds, real_label_preds = model.discriminator(real_data)
        real_loss = loss_criterion(real_preds, ones)
        clf_loss = clf_loss_criterion(real_label_preds, real_labels)
        # get generator output and discriminator preds on it
        fake_data = model.generator(latent_vectors)
        # detach fake data from computation graph when feeding to discriminator
        # because generator doesn't need to be updated in this step
        fake_preds, _ = model.discriminator(fake_data.detach())
        # discriminator loss on fake images
        fake_loss = loss_criterion(fake_preds, zeros)
        # total discriminator loss
        alpha = 1
        d_loss = 1*(real_loss + fake_loss) + alpha*clf_loss
        total_d_loss += d_loss.item()
        # compute all gradients
        d_loss.backward()
        # clear out gradients for generator. Generator shouldn't help discriminator
        d_optimizer.step()

        gen_optimizer.zero_grad()
        # compute loss for generator using discriminator's output but generator's expectation
        fake_preds, _ = model.discriminator(fake_data)
        beta = 1
        g_loss = beta * loss_criterion(fake_preds, ones)
        g_loss.backward()
        total_gen_loss += g_loss.item()
        gen_optimizer.step()
    avg_gen_loss = total_gen_loss/(batch_idx+1)
    avg_d_loss = total_d_loss/(batch_idx+1)
    avg_total_loss = avg_gen_loss + avg_d_loss
    print(f"Train epoch: {epoch} [{(batch_idx+1) * len(real_data)}/{len(train_loader.dataset)}]\
          \tGenerator Loss: {avg_gen_loss:.6f}\tDiscriminator Loss: {avg_d_loss:.6f}")
    return avg_gen_loss, avg_d_loss, avg_total_loss


def main():
    """
    main program starts here
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 GAN")
    parser.add_argument('--input-channels', type=int, default=3, metavar='IN',
                        help='input channels in the images. (default: 3)')
    parser.add_argument('--n-classes', type=int, default=10, metavar='C',
                        help='no. of classes. (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--gen-lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate for generator (default: 1e-4)')
    parser.add_argument('--d-lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate for discriminator (default: 1e-4)')
    parser.add_argument('--nf', type=int, default=32, metavar='NF',
                        help='no. of filters (default: 32)')
    parser.add_argument('--latent-dim', type=int, default=100, metavar='LD',
                        help='size of latent dimension (default: 128)')
    parser.add_argument('--gamma', type=float, default=1e-4, metavar='M',
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

    # set device for model
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

    model = GAN(args.n_classes, device, args.latent_dim).to(device)
    # opt_model = torch.compile(model)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_criterion = nn.BCELoss()
    clf_loss_criterion = nn.CrossEntropyLoss()
    summary(model)

    print_training_parameters(args, device)

    results_dir = root_dir+"/GAN_cifar10/results"
    models_dir = results_dir+"/saved_models"
    inference_dir = results_dir+"/generated"

    # Run training loops
    losses = defaultdict(list)
    for epoch in range(1,args.epochs+1):
        avg_gen_loss, avg_d_loss, avg_total_loss = train_model(model, device, train_loader, gen_optimizer, d_optimizer, loss_criterion, clf_loss_criterion, epoch)
        losses['gen_loss'].append(avg_gen_loss)
        losses['d_loss'].append(avg_d_loss)
        losses['total_loss'].append(avg_total_loss)
        # generate a batch of images with fixed noise to evaluate generator
        model.eval()
        z = model.generator.sample_latent_vectors(n_samples=50)
        # z = model.generator.eval_noise
        generate_images(5,10,model, inference_dir, device, z=z, name=f"epoch_{epoch}")
        # scheduler.step()
        if (args.dry_run):
            break

    save_gan_plots(losses, results_dir, name="plot")
    # save model
    model_name = f"GAN_CIFAR10"
    if (args.save_model):
        save_model(model, models_dir, name=model_name)

    # generate samples using the model
    model.eval()
    generate_images(5,10,model, inference_dir, device, name="sample")

if __name__ == '__main__':
    main()