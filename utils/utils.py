from torchinfo import summary
import matplotlib.pyplot as plt
import pathlib
import torch
from time import time

# A util function to print model summary
def print_model_summary(model):
    print(summary(model.conv, input_size=(1,28,28)))
    print(summary(model.dense, input_size=(1, 4*model.nf * 3*3)))

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toatal model params: {total_params}, Trainable params: {trainable_params}")
    return total_params, trainable_params

def save_plots(losses, dir, name=None):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    if not name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_plot_path = "plot-"+timestr
    else:
        save_plot_path = name
    save_plot_path = f"{dir}/{save_plot_path}"
    epochs = len(losses['train_loss'])
    plt.plot(range(1,epochs+1), losses['train_loss'], label='Train loss')
    plt.plot(range(1,epochs+1), losses['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"MNIST Losses")
    plt.savefig(save_plot_path, facecolor='w', edgecolor='none')
    #plt.show()

def save_model(model, dir, name=None):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    if not name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_model_path = "/MNIST-"+timestr+".pth"
    else:
        save_model_path = name+".pth"
    save_model_path = f"{dir}/{save_model_path}"
    torch.save(model.state_dict(), save_model_path)

# load model
def load_model(model_name, model_class, dir, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model_path = dir+"/"+model_name+".pth"
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except OSError as e:
        print(f"{e.strerror}: {e.filename}")
        return None
    return model