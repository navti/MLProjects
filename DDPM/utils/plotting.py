import pathlib
import matplotlib.pyplot as plt
import time
import torch
import math

plt.switch_backend("agg")


def save_plots(losses, results_dir, name=None):
    """
    Save loss and score plots
    :param losses: dict with losses
    :param results_dir: directory where plots will be saved
    :param name: plot file name
    """
    fig = plt.figure()
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_plot_path = results_dir + "/plot-" + timestr
    else:
        save_plot_path = results_dir + "/" + name
    if "train" in losses:
        plt.plot(losses["train"][1], losses["train"][0], label="Train loss")
        plt.plot(losses["validation"][1], losses["validation"][0], label="Val loss")
        plt.ylabel("Loss")
        plt.title(f"DDPM losses")
        plt.ylim(0, 5)
    else:
        plt.plot(losses["fid"][1], losses["fid"][0], label="FID score")
        plt.ylabel("Score")
        plt.title(f"DDPM FID Scores")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig(save_plot_path, facecolor="w", edgecolor="none")
    plt.close(fig)
    # print(f"Plot saved at: {save_plot_path}")


def draw(xhat, ts, inference_dir=None, name=None):
    """
    draw noisy samples
    :param rows: rows of images in the figure
    :param cols: columns of images in the figure
    :param inference_dir: directory where the sample images to store
    :param device: device where model should be run
    :param z: the noise vector to use with generator, if none, one will be created
    :param name: name of figure to save
    :returns: None
    """
    proj_dir = "/".join(__file__.split("/")[:-2])
    results_dir = f"{proj_dir}/results"
    if not inference_dir:
        inference_dir = results_dir
    pathlib.Path(inference_dir).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not name:
        save_fig_path = inference_dir + "/noisy_samples_" + timestr
    else:
        save_fig_path = inference_dir + "/" + name
    batch_size = xhat.shape[0]
    rows = cols = int(math.sqrt(batch_size))
    size = (rows * cols, 3, 32, 32)
    xhat = torch.einsum("nchw->nhwc", xhat)
    xhat = xhat.view(rows, cols, *xhat.shape[1:]).detach().cpu()
    ts = ts.view(rows, cols, *ts.shape[1:]).detach().cpu()
    # print(xhat.shape)
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    for A, I, T in zip(axs, xhat, ts):
        for ax, img, t in zip(A, I, T):
            ax.set_aspect("equal")
            ax.set_title(f"t = {t.item()}", size="medium")
            ax.axis("off")
            ax.imshow(img)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.5)
    plt.savefig(save_fig_path, facecolor="w", edgecolor="none")
    plt.close(fig)
