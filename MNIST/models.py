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
    