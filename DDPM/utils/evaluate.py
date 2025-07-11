import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plotting import *
import pathlib
import time

plt.switch_backend("agg")


def generate_images(rows, cols, model, diffuser, inference_dir, device, name=None):
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
        save_fig_path = inference_dir + "/samples_" + timestr
    else:
        save_fig_path = inference_dir + "/" + name

    size = (rows * cols, 3, 32, 32)
    xhat = diffuser.sample(model, size, device)
    xhat = torch.einsum("nchw->nhwc", xhat)
    xhat = xhat.view(rows, cols, *xhat.shape[1:]).detach().cpu()
    # print(xhat.shape)
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    for A, I in zip(axs, xhat):
        for ax, img in zip(A, I):
            ax.set_aspect("equal")
            ax.axis("off")
            ax.imshow(img)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_fig_path, facecolor="w", edgecolor="none")
    plt.close(fig)


def validate_model(model, val_loader, device, loss_criterion):
    model.eval()
    total_loss = 0
    for batch_idx, (xt, eps, t_embs, ts, labels) in enumerate(val_loader):
        xt = xt.to(device)
        eps = eps.to(device)
        t_embs = t_embs.to(device)
        labels = labels.to(device)
        batch_size = xt.shape[0]
        predicted_eps = model(xt, t_embs)
        loss = loss_criterion(predicted_eps, eps)
        avg_step_loss = 1e4 * loss.item() / batch_size
        total_loss += avg_step_loss
    return total_loss / (batch_idx + 1)
