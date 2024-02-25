import torch
import argparse
import glob
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils import *
from models import *

def train_model(args, model, device, train_loader, loss_criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    total = 0
    total_acc = 0
    for batchidx, (data, target) in enumerate(train_loader):
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
    print(f"Train epoch: {epoch} [{batchidx * args.batch_size}/{len(train_loader.dataset)}\
          ({(100. * batchidx/len(train_loader)):.0f}%)]\tLoss: {avg_loss:.6f}\tAcc: {100. * avg_acc:.6f}%")
    return 100. * avg_acc

def test_model(args, model, device, test_loader, loss_criterion):
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
    return 100. * correct/total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.25, metavar='LR',
                        help='learning rate (default: 0.5)')
    parser.add_argument('--nf', type=int, default=32, metavar='NF',
                        help='no. of filters (default: 32)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
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

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    data_dir = "data/"
    train_files = glob.glob('**/*_batch_*', recursive=True)
    test_files = glob.glob('**/*test_batch*', recursive=True)
    train_files = list(map(unpickle, train_files))
    test_files = list(map(unpickle, test_files))
    test = combine_dicts(test_files)
    train = combine_dicts(train_files)
    print(f"test data shape: {test['data'].shape}")
    print(f"train data shape: {train['data'].shape}")

    # make train and test dataset
    trainset = make_dataset(train)
    testset = make_dataset(test)
    # data loaders
    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    model = MODEL(3,args.nf).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_criterion = nn.CrossEntropyLoss()
    print_model_summary(model)

    train_acc = []
    test_acc = []
    for epoch in range(1,args.epochs+1):
        train_acc.append(train_model(args, model, device, train_loader, loss_criterion, optimizer, epoch))
        test_acc.append(test_model(args, model, device, test_loader, loss_criterion))
        if (args.dry_run):
            break
    save_plot(train_acc, test_acc)

    if (args.save_model):
        save_model(model)

if __name__ == '__main__':
    main()