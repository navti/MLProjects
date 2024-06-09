from torch import nn
import torch.nn.functional as F
from torchinfo import summary

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
    def __init__(self, nf):
        super(FCNN, self).__init__()
        # in: nfx4x4, flatten and feed to linear layer
        self.fc1 = nn.Linear(4*nf*4*4, 4*nf)
        self.bn_d1 = nn.BatchNorm1d(4*nf)
        self.fc2 = nn.Linear(4*nf, nf)
        self.bn_d2 = nn.BatchNorm1d(nf)        
        self.fc3 = nn.Linear(nf, 10)
        self.bn_d3 = nn.BatchNorm1d(10)
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


# Conv block used in Resnet model
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# Resnet9
class RESNET9(nn.Module):
    def __init__(self, in_channels, nf, num_classes):
        super(RESNET9, self).__init__()
        self.nf = nf
        self.conv1 = conv_block(in_channels, nf)
        self.conv2 = conv_block(nf, 2*nf, pool=True)
        self.res1 = nn.Sequential(conv_block(2*nf, 2*nf), conv_block(2*nf, 2*nf))

        self.conv3 = conv_block(2*nf, 4*nf, pool=True)
        self.conv4 = conv_block(4*nf, 8*nf, pool=True)
        self.res2 = nn.Sequential(conv_block(8*nf, 8*nf), conv_block(8*nf, 8*nf))

        # classifier head
        self.fc = nn.Sequential(nn.MaxPool2d(kernel_size=4),
                                nn.Flatten(),
                                nn.Linear(8*nf, num_classes))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        return self.fc(x)

    def model_summary(self):
        return summary(self, input_size=(3,32,32))
