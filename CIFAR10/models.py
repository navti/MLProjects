from torch import nn
import torch.nn.functional as F

# conv layers of the network
class CNN(nn.Module):
    def __init__(self, channels, nf):
        super(CNN, self).__init__()
        # in: 3x32x32, out: nfx32x32
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nf, kernel_size=3, stride=1, padding='same')
        # in: 32x32, out: 16x16
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # in: nfx16x16, out: nfx16x16
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding='same')
        # apply maxpool again
        # in: nfx8x8, out: nfx8x8
        self.conv3 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding='same')
        #apply maxpool again
        self.bn_conv = nn.BatchNorm2d(nf)
        self.dropout = nn.Dropout(0.25)

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
    def __init__(self, nf):
        super(FCNN, self).__init__()
        # in: nfx4x4, flatten and feed to linear layer
        self.fc1 = nn.Linear(nf*4*4, 100)
        self.bn_d1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        self.bn_d2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 10)
        self.bn_d3 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn_d1(x)
        x = self.dropout(x)
        #x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn_d2(x)
        x = self.dropout(x)
        #x = F.relu(x)
        
        x = self.fc3(x)
        x = self.bn_d3(x)

        return x

# combined model
class MODEL(nn.Module):
    def __init__(self, channels, nf):
        super(MODEL, self).__init__()
        self.conv = CNN(channels, nf)
        self.dense = FCNN(nf)
        self.nf = nf
        self.channels = channels
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.dense(x)
        return x