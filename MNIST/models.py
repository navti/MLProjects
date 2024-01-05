import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt 
import pathlib
from time import time

# conv layers of the network
class CNN(nn.Module):
    def __init__(self, channels, nf):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nf, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=2*nf, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=2*nf, out_channels=4*nf, kernel_size=3, stride=1, padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.bn_conv = nn.BatchNorm2d(nf)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn_conv(x)
        #x = self.dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        #x = self.bn_conv(x)
        #x = self.dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)

        x = self.conv3(x)
        #x = self.bn_conv(x)
        #x = self.dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)

        return x

# dense layers of the network
class FCNN(nn.Module):
    def __init__(self, nf, num_classes):
        super(FCNN, self).__init__()
        # in: nfx4x4, flatten and feed to linear layer
        self.fc1 = nn.Linear(4*nf*3*3, 4*nf)
        self.bn_d1 = nn.BatchNorm1d(4*nf)
        self.fc2 = nn.Linear(4*nf, nf)
        self.bn_d2 = nn.BatchNorm1d(nf)        
        self.fc3 = nn.Linear(nf, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn_d1(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn_d2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc3(x)
        #x = self.bn_d3(x)
        return x

# combined model
class MODEL(nn.Module):
    def __init__(self, channels, nf, num_classes):
        super(MODEL, self).__init__()
        self.conv = CNN(channels, nf)
        self.dense = FCNN(nf, num_classes)
        self.nf = nf
        self.channels = channels
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.dense(x)
        return x
    

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