import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class FCNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=num_classes, kernel_size=5, stride=1, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = F.relu(x)
        x = self.pool(self.conv2(x))
        x = F.relu(x)
        x = self.pool(self.conv3(x))
        x = F.relu(x)
        x = self.pool(self.conv4(x))
        x = F.relu(x)
        x = x.reshape(x.shape[0],-1)
        return x


def train_model(model, train_loader, num_epochs, optimizer, criterion, device):
    for epoch in range(num_epochs):
        model.train()
        for batch, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        num_correct, num_total, acc = get_accuracy(model, train_loader, device)
        print(f"Training Accuracy: Epoch: {epoch+1}, Accuracy: {num_correct}/{num_total} = {acc*100}%")
    
def test_model(model, test_loader, device):
    model.eval()
    num_correct, num_total,  acc = get_accuracy(model,test_loader, device)
    model.train()
    print(f"Testing Accuracy: {num_correct}/{num_total} = {acc*100}%")

def get_accuracy(model, data_loader, device):
    num_correct = 0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            predictions = scores.argmax(axis=1)
            num_correct += (predictions==targets).sum()
            num_total += len(predictions)
    acc = num_correct/num_total
    model.train()
    return num_correct, num_total, acc

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toatal model params: {total_params}, Trainable params: {trainable_params}")
    return total_params, trainable_params

batch_size = 64
train_set = datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.MNIST(root='data',train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

num_classes = 10
learning_rate = 0.01
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
inp_channels = 1
ksize = 3
model = CNN(inp_channels, num_classes)
count_model_params(model)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
train_model(model, train_loader,num_epochs,optimizer,criterion,device)
test_model(model, test_loader, device)
