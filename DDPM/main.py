import torch
from torch import optim
from torch import nn
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from model import *
import sys
from collections import defaultdict
from utils.data_loading import make_cifar_set
from diffuser import GaussianDiffuser
from torch.utils.data import DataLoader
from utils.evaluate import *
from args import *

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
    print(f"LR scheduler step size: {args.lr_steps}")
    print(f"Gamma for stepLR scheduler: {args.gamma}")
    print(f"No. of time steps: {args.time_steps}")
    print(f"Latent dimension: {args.d_model}")
    print(f"Beta start: {args.beta_start}")
    print(f"Beta end: {args.beta_end}")
    print(f"===========================================")
    print(f"")


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

    results_dir = f"{proj_dir}/results"
    models_dir = f"{proj_dir}/model/saved_models"
    inference_dir = results_dir + "/generated"

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
        scheduler = StepLR(optimizer, step_size=args.lr_steps, gamma=args.gamma)
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
            if args.enable_steplr and epoch <= 90:
                scheduler.step()
            if epoch % args.epoch_steps == 0:
                save_plots(losses, results_dir, name="losses")
                generate_images(
                    5,
                    5,
                    model,
                    gaussian_diffuser,
                    inference_dir,
                    device,
                    name=f"samples_epoch_{epoch}",
                )
            if args.dry_run:
                break
        # save model
        model_name = f"ddpm"
        if args.save_model:
            save_model(model, models_dir, name=model_name)

    else:
        # enable block checkpointing
        # model_kwargs["checkpointing"] = True
        model = load_model(args.load, **model_kwargs)
        if model == None:
            print("Check if the correct model path was provided to load from.")
            parser.print_help(sys.stderr)
            sys.exit(1)
        betas = [args.beta_start, args.beta_end]
        T = args.time_steps
        gaussian_diffuser = GaussianDiffuser(betas=betas, T=T)
        model = model.to(device)
        generate_images(
            2, 5, model, gaussian_diffuser, inference_dir, device, name=f"samples"
        )


if __name__ == "__main__":
    main()
