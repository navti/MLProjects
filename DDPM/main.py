import torch
from torch import optim
from torch import nn
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from model import *
import argparse
from collections import defaultdict
from utils.data_loading import make_cifar_set
from diffuser import GaussianDiffuser
from torch.utils.data import DataLoader
from utils.evaluate import *

torch.set_float32_matmul_precision("high")


def print_training_parameters(args, device):
    """
    print training parameters of the model
    :param args: args containing params
    :param device: device being used to host the model
    :returns: None
    """
    print(f"")
    print(f"=========== Training Parameters ===========")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"No. of time steps: {args.time_steps}")
    print(f"Latent dimension: {args.d_model}")
    print(f"Beta start: {args.beta_start}")
    print(f"Beta end: {args.beta_end}")
    print(f"===========================================")
    print(f"")


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 DDPM")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=3,
        metavar="IN",
        help="input channels in the images. (default: 3)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=10,
        metavar="C",
        help="no. of classes. (default: 10)",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=1000,
        metavar="T",
        help="no. of time steps. (default: 1000)",
    )
    parser.add_argument(
        "--beta-start",
        metavar="B1",
        type=float,
        default=1e-4,
        help="Beta schedule start value",
    )
    parser.add_argument(
        "--beta-end",
        metavar="B2",
        type=float,
        default=0.02,
        help="Beta schedule end value",
    )
    parser.add_argument(
        "--lr",
        "-l",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        metavar="NUM_EPOCHS",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="BATCH_SIZE",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        default=False,
        help="Train model with given parameters.",
    )
    parser.add_argument(
        "--nf", type=int, default=8, metavar="NF", help="no. of filters (default: 32)"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        metavar="LD",
        help="size of latent dimension (default: 128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--load",
        "-f",
        metavar="MODEL_PATH",
        type=str,
        default=None,
        help="Load model from a .pth file. Required when running inference.",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=False,
        help="Use mixed precision",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use mixed precision"
    )
    return parser, parser.parse_args()


def main():
    """
    main program starts here
    """

    proj_dir = "/".join(__file__.split("/")[:-1])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    parser, args = get_args()
    torch.manual_seed(args.seed)
    args.train = True

    model_kwargs = {}
    model_kwargs["n_channels"] = args.n_channels
    model_kwargs["n_classes"] = args.n_classes
    model_kwargs["d_model"] = args.d_model
    model_kwargs["nf"] = args.nf
    model_kwargs["checkpointing"] = args.checkpointing
    model_kwargs["bilinear"] = args.bilinear

    if args.train:
        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.batch_size}
        validation_kwargs = {"batch_size": args.batch_size}
        if use_cuda:
            cuda_kwargs = {
                "num_workers": 1,
                "pin_memory": True,
                "shuffle": True,
                "drop_last": True,
            }
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
            validation_kwargs.update(cuda_kwargs)

        cifar_data_dir = f"{proj_dir}/data"
        betas = [args.beta_start, args.beta_end]
        T = args.time_steps
        gaussian_diffuser = GaussianDiffuser(betas=betas, T=T)
        trainset = make_cifar_set(data_dir=cifar_data_dir, diffuser=gaussian_diffuser)
        train_loader = DataLoader(trainset, **train_kwargs)

        if args.load != None:
            print(f"Training existing model.")
            print(f"Loading model from the given path: {args.load}")
            model = load_model(args.load, **model_kwargs).to(device)
        else:
            model = UNet(**model_kwargs).to(device)

        loss_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        grad_scaler = torch.amp.GradScaler(enabled=True)
        print("Model summary:")
        summary(model)

        print_training_parameters(args, device)

        # Run training loop
        losses = defaultdict(list)
        for epoch in range(1, args.epochs + 1):
            # train
            avg_epoch_loss = train(
                model,
                device,
                epoch,
                optimizer,
                train_loader,
                loss_criterion,
                grad_scaler,
            )
            losses["train"].append(avg_epoch_loss)
            # scheduler.step()
            if args.dry_run:
                break

        results_dir = f"{proj_dir}/results"
        models_dir = f"{proj_dir}/model/saved_models"
        inference_dir = results_dir + "/generated"

        save_plots(losses, results_dir, name="losses")

        # save model
        model_name = f"ddpm"
        if args.save_model:
            save_model(model, models_dir, name=model_name)

        generate_images(
            2, 5, model, gaussian_diffuser, inference_dir, device, name=f"samples"
        )


if __name__ == "__main__":
    main()
