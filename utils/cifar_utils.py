import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import time
import torch.nn.functional as F
from torchinfo import summary

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return {'data': dict[b'data'], 'labels': dict[b'labels']}

def combine_dicts(dicts: list)->dict:
    # keys: data, labels
    n = len(dicts)
    data, labels = dicts[0]['data'], dicts[0]['labels']
    for i in range(1,n):
        data = np.concatenate([data, dicts[i]['data']], axis=0)
        labels = labels + dicts[i]['labels']
    return {'data':data, 'labels':labels}

def array_to_rgb(arr: np.ndarray)->np.ndarray:
    # take an 1x3072
    # reshape to rgb: 32, 32, 3
    rgb = np.reshape(arr, (32, 32, 3), order='F').astype(dtype=np.float32)
    rgb = rgb / 255.
    # rotate by 90 degrees
    return np.rot90(rgb, k=1, axes=(1,0))

def show_image(arr: np.ndarray)->None:
    rgb_img = array_to_rgb(arr)
    print(rgb_img.shape)
    plt.figure(figsize=(2,2))
    plt.imshow(rgb_img)

def torch_from_numpy(arr: list)->torch.tensor:
    # convert to torch tensors
    #ndarrays = list(map(array_to_rgb, arr))
    tensors = [torch.from_numpy(item.copy()) for item in arr]
    return torch.einsum('nhwc->nchw', torch.stack(tensors, dim=0))

def make_dataset(d: dict)->list:
    data = list(map(array_to_rgb, d['data']))
    tensors = [torch.from_numpy(item.copy()) for item in data]
    tensors = torch.einsum('nhwc->nchw', torch.stack(tensors, dim=0))
    return list(zip(tensors, d['labels']))

def print_model_summary(model):
    summary(model.conv, input_size=(3,32,32))
    summary(model.dense, input_size=(1, model.nf * 16))

def save_model(model, models_dir, name=None):
    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_model_path = models_dir+"/CIFAR10-"+timestr+".pth"
    else:
        save_model_path = models_dir+"/"+name+".pth"
    torch.save(model.state_dict(), save_model_path)

# load model
def load_model(model_name, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model_path = "results/saved_models/"+model_name+".pth"
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except OSError as e:
        print(f"{e.strerror}: {e.filename}")
        return None
    return model

def save_plot(train_acc, test_acc, results_dir, name=None):
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = results_dir+"/plot-"+timestr
    else:
        save_plot_path = results_dir+"/"+name
    epochs = len(train_acc)
    plt.plot(range(1,epochs+1), train_acc, label='Train')
    plt.plot(range(1,epochs+1), test_acc, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_plot_path, facecolor='w', edgecolor='none')

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

def elbo_loss(xhat, x, mu, logvar, beta=1):
    batch_size = xhat.shape[0]
    recon_loss = F.mse_loss(xhat, x, reduction='sum') / batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),axis=1)
    kl_loss = kl_div.mean()
    beta = beta #kl_loss/recon_loss
    total_loss = recon_loss + beta  * kl_loss
    return total_loss, recon_loss, kl_loss, beta

def save_plots(losses, beta, results_dir, name=None):
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = results_dir+"/plot-"+timestr
    else:
        save_plot_path = results_dir+"/"+name
    epochs = len(losses['kl_loss'])
    plt.plot(range(1,epochs+1), losses['total_loss'], label='Total loss')
    plt.plot(range(1,epochs+1), losses['recon_loss'], label='Reconstruction loss')
    plt.plot(range(1,epochs+1), losses['kl_loss'], label='KL loss')
    #plt.plot(range(1,epochs+1), losses['cl_loss'], label='CL loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"VAE CIFAR10 beta = {beta}")
    plt.savefig(save_plot_path, facecolor='w', edgecolor='none')
    #plt.show()

def generate_samples(rows, cols, model, inference_dir):
    pathlib.Path(inference_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_fig_path = inference_dir+"/samples_"+timestr
    latent_dim = model.latent_dim
    z = model.encoder.N.sample([rows*cols,latent_dim])
    xhat = model.decoder(z)
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
    #plt.show()