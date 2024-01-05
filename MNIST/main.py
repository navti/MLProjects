import torch
import argparse
from torch import optim
from torch.optim.lr_scheduler import StepLR
from models import *
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import defaultdict
import sys

# add to path variable so module can be found
root_dir = '/'.join(__file__.split('/')[:-2])
sys.path.append(root_dir)

def train_model(model, device, train_loader, loss_criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    total = 0
    total_acc = 0
    for batchidx, (data, target) in enumerate(train_loader):
        #print(f'batch id: {batchidx}')
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        predictions = output.argmax(dim=1, keepdim=True)
        #print(f'predictions: {predictions.shape}, target: {target.shape}')
        correct = predictions.eq(target.view_as(predictions)).sum().item()
        total_acc += (correct/len(data))
        total += len(data)
    avg_acc = total_acc/len(train_loader)
    avg_loss = train_loss/total
    print(f"Train epoch: {epoch} [{(batchidx+1) * len(data)}/{len(train_loader.dataset)}\
          ({(100. * batchidx/len(train_loader)):.0f}%)]\tLoss: {avg_loss:.6f}\tAcc: {100. * avg_acc:.6f}%")
    return 100. * avg_acc, avg_loss

def test_model(model, device, test_loader, loss_criterion):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_criterion(output, target).item()
            predictions = output.argmax(dim=1, keepdim=True)
            #print(f'predictions: {predictions.shape}, target: {target.shape}')
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += len(data)
    test_loss /= total
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * correct/total:.0f}%)")
    return 100. * correct/total, test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--nf', type=int, default=16, metavar='NF',
                        help='no. of filters (default: 16)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='NC',
                        help='no. of classes (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.01, metavar='M',
                        help='Learning rate step gamma (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'using device: {device}')
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data_dir = root_dir+"/data/"
    batch_size = args.batch_size
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomInvert()])
    train_set = datasets.MNIST(root=data_dir,train=True,transform=transform,download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_set = datasets.MNIST(root=data_dir,train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    num_classes = args.num_classes
    num_filters = args.nf
    learning_rate = args.lr
    num_epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_criterion = nn.CrossEntropyLoss()
    inp_channels = 1
    model = MODEL(inp_channels, num_filters, num_classes).to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)

    losses = defaultdict(list)

    for epoch in range(1,num_epochs+1):
        _, tloss = train_model(model, device, train_loader, loss_criterion, optimizer, epoch)
        losses['train_loss'].append(tloss)
        _, vloss = test_model(model, device, test_loader, loss_criterion)
        losses['val_loss'].append(vloss)
        scheduler.step()
        if (args.dry_run):
            break

    results_dir = root_dir+"/MNIST/results"
    models_dir = results_dir+"/saved_models"

    # save plots
    save_plots(losses, results_dir, 'mnist_loss')

    #save model
    if (args.save_model):
        save_model(model, models_dir, 'mnist_sample_model')

if __name__ == '__main__':
    main()