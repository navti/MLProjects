import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import time
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

def save_model(model):
    pathlib.Path("results/saved_models").mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_model_path = "results/saved_models/CIFAR10-"+timestr+".pth"
    torch.save(model.state_dict(), save_model_path)

def save_plot(train_acc, test_acc, name=None):
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = "results/plot-"+timestr
    else:
        save_plot_path = "results/"+name
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