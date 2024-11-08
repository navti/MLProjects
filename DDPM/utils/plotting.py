import pathlib
import matplotlib.pyplot as plt
import time


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
    epochs = len(losses["train"] if "train" in losses else losses["bleu"])
    if "train" in losses:
        plt.plot(range(1, epochs + 1), losses["train"], label="Train loss")
        # plt.plot(range(1, epochs + 1), losses["validation"], label="Validation loss")
        plt.ylabel("Loss")
    else:
        plt.plot(range(1, epochs + 1), losses["bleu"], label="BLEU score")
        plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(f"DDPM loss curve")
    plt.savefig(save_plot_path, facecolor="w", edgecolor="none")
    print(f"Plot saved at: {save_plot_path}")
