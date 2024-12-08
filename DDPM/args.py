import argparse


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
        help="no. of time steps for diffusion (default: 1000)",
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
        default=2e-4,
        help="Peak Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr-schedule",
        metavar="NAME",
        type=str,
        default="constant",
        help="learning rate schedule ['constant', 'linear', 'step', 'cosine']",
    )
    parser.add_argument(
        "--steplr-steps",
        metavar="S",
        type=int,
        default=100000,
        help="Steps period for step LR",
    )
    parser.add_argument(
        "--steplr-factor",
        metavar="F",
        type=float,
        default=0.5,
        help="Factor to reduce LR by in step LR",
    )
    parser.add_argument(
        "--total-steps",
        metavar="TS",
        type=int,
        default=200_000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--annealing-period",
        metavar="P",
        type=int,
        default=None,
        help="Number of training steps after which LR annealing schedule repeats",
    )
    parser.add_argument(
        "--warmup-steps",
        metavar="WARMUP_STEPS",
        type=int,
        default=10_000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--logging-steps",
        metavar="LOG_STEPS",
        type=int,
        default=1000,
        help="Number of steps after which loss curve is updated and images are sampled",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="BATCH_SIZE",
        type=int,
        default=256,
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
        "--train-subset",
        action="store_true",
        default=False,
        help="Train model with 1% of dataset.",
    )
    parser.add_argument(
        "--nf",
        type=int,
        default=32,
        metavar="NF",
        help="no. of base filters/channels (default: 32)",
    )
    parser.add_argument(
        "--t-dim",
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
        metavar="NAME",
        type=str,
        default=None,
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
        "--checkpoint",
        type=int,
        default=None,
        help="No. of steps for model checkpointing during training.",
    )
    parser.add_argument(
        "--dropout",
        metavar="D",
        type=float,
        default=0.1,
        help="Dropout factor to be used in the model layers.",
    )
    return parser, parser.parse_args()
