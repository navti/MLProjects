import torch
from torch import optim
from torch import nn
from torchinfo import summary
from torch.optim.lr_scheduler import LambdaLR
from model import *
import sys
import os
import time
from collections import defaultdict
from utils.data_loading import make_cifar_set
from diffuser import GaussianDiffuser
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import random_split
from utils.evaluate import *
from args import *
from utils.metrics_fid import fid_score

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
    print(f"Training steps: {args.total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Peak Learning Rate: {args.lr}")
    print(f"No. of diffusion time steps: {args.time_steps}")
    print(f"Time embedding dimension: {args.t_dim}")
    print(f"Beta start: {args.beta_start}")
    print(f"Beta end: {args.beta_end}")
    print(f"===========================================")
    print(f"")


if __name__ == "__main__":

    parser, args = get_args()

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / args.warmup_steps
        elif args.lr_schedule == "constant":
            return 1.0
        elif args.lr_schedule == "step":
            f = step // args.steplr_steps
            return args.steplr_factor**f
        else:
            progress = float(step - args.warmup_steps) / float(
                args.total_steps - args.warmup_steps
            )
            if args.lr_schedule == "linear":
                return max(1e-5, 1 - progress)
            return max(1e-6, (0.5 * (1.0 + math.cos(math.pi * progress))))

    """
    main program starts here
    """

    proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    model_kwargs = {}
    model_kwargs["n_channels"] = args.n_channels
    model_kwargs["n_classes"] = args.n_classes
    model_kwargs["t_dim"] = args.t_dim
    model_kwargs["nf"] = args.nf
    model_kwargs["bilinear"] = args.bilinear

    results_dir = f"{proj_dir}/results"
    models_dir = f"{proj_dir}/model/saved_models"
    checkpoint_dir = f"{proj_dir}/model/checkpoints"
    inference_dir = results_dir + "/generated"
    # args.train = True
    if args.train or args.train_subset:
        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.batch_size}
        validation_kwargs = {"batch_size": args.batch_size}
        if use_cuda:
            cuda_kwargs = {
                "num_workers": 2 * os.cpu_count(),
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
        train_set, test_set = make_cifar_set(
            data_dir=cifar_data_dir, diffuser=gaussian_diffuser
        )
        trainset, validation_set = random_split(train_set, [0.9, 0.1])
        if args.train_subset:
            subset_size = int(0.01 * len(trainset))
            indices = torch.arange(len(trainset))
            subset_indices = indices[:subset_size]
            subset_sampler = SubsetRandomSampler(subset_indices)
            train_kwargs.update({"sampler": subset_sampler, "shuffle": False})

        # data loaders
        train_loader = DataLoader(trainset, **train_kwargs)
        val_loader = DataLoader(validation_set, **validation_kwargs)
        test_loader = DataLoader(test_set, **test_kwargs)
        model = UNet(**model_kwargs).to(device)
        # model = torch.compile(model, mode="default", fullgraph=False)
        loss_criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = LambdaLR(optimizer, lr_lambda)
        grad_scaler = torch.amp.GradScaler(enabled=True)

        losses = {"train": [[], []]}
        losses.update({"validation": [[], []]})
        lr_schedule = {"lr": [[], []]}
        # load checkpointed model
        if args.load != None:
            print(f"Training existing model.")
            print(f"Loading model from the given path: {args.load}")
            losses, lr_schedule = load_checkpoint(
                model, args.load, optimizer, scheduler, device
            )
            # load peak lr from args
            scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = args.lr
            # model = load_model(args.load, **model_kwargs).to(device)
        print("Model summary:")
        summary(model)

        print_training_parameters(args, device)

        # Run training loop
        current_step = len(losses["train"][0])
        # train
        while current_step < args.total_steps:
            # start = time.time()
            for batch_idx, (xt, eps, t_embs, ts, labels) in enumerate(train_loader):
                # print(f"time: {(time.time() - start)*10000}")
                model.train()
                xt = xt.to(device)
                eps = eps.to(device)
                t_embs = t_embs.to(device)
                labels = labels.to(device)
                batch_size = xt.shape[0]
                # use automatic mixed precision training
                with torch.autocast(device.type, enabled=True):
                    predicted_eps = model(xt, t_embs)
                    loss = loss_criterion(predicted_eps, eps)
                    avg_step_loss = 1e4 * loss.item() / batch_size
                optimizer.zero_grad()
                # scale gradients when doing backward pass, avoid vanishing gradients
                grad_scaler.scale(loss).backward()
                # unscale gradients before applying
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

                current_step += 1
                losses["train"][0].append(avg_step_loss)
                losses["train"][1].append(current_step)
                lr_schedule["lr"][0].append(scheduler.get_last_lr()[0])
                lr_schedule["lr"][1].append(current_step)
                if current_step % args.logging_steps == 0:
                    val_loss = validate_model(model, val_loader, device, loss_criterion)
                    losses["validation"][0].append(val_loss)
                    losses["validation"][1].append(current_step)

                    # fid = fid_score(test_set, model, gaussian_diffuser, device)
                    # losses["fid"][0].append(fid)
                    # losses["fid"][1].append(current_step)

                    print(
                        f"Step {current_step}/{args.total_steps}:\n\tTrain Loss: {avg_step_loss:.5f}, \tVal Loss: {val_loss:.5f}"
                    )
                    print(
                        f"\tStep: {current_step}, LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    print(f"=========================")
                    save_plots(losses, results_dir, name="losses")
                    save_plots(lr_schedule, results_dir, name="lrschedule")
                    generate_images(
                        10,
                        10,
                        model,
                        gaussian_diffuser,
                        inference_dir,
                        device,
                        name=f"samples_step_{current_step}",
                    )
                if args.checkpoint != None and current_step % args.checkpoint == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        losses,
                        lr_schedule,
                        checkpoint_dir,
                        scheduler,
                        filename=f"checkpoint_{current_step}",
                    )
                if current_step >= args.total_steps:
                    break
                # start = time.time()
            if args.dry_run:
                break

        # save model
        model_name = f"ddpm"
        if args.save_model != None:
            save_checkpoint(
                model,
                optimizer,
                losses,
                lr_schedule,
                models_dir,
                scheduler,
                filename=f"{model_name}",
            )

    else:
        # model = load_model(args.load, **model_kwargs)
        model = UNet(**model_kwargs).to(device)
        if args.load == None:
            print("Check if the correct model path was provided to load from.")
            parser.print_help(sys.stderr)
            sys.exit(1)
        load_checkpoint(model, args.load, optimizer=None, scheduler=None, device=device)
        betas = [args.beta_start, args.beta_end]
        T = args.time_steps
        gaussian_diffuser = GaussianDiffuser(betas=betas, T=T)
        model = model.to(device)
        generate_images(
            16, 16, model, gaussian_diffuser, inference_dir, device, name=f"samples"
        )
