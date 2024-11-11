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
        "--gamma",
        type=float,
        default=1e-1,
        metavar="G",
        help="Learning rate factor gamma (default: 0.1)",
    )
    parser.add_argument(
        "--lr-steps",
        metavar="STEP_SIZE",
        type=int,
        default=50,
        help="Step size for step lr scheduler",
    )
    parser.add_argument(
        "--enable-steplr",
        action="store_true",
        default=False,
        help="Enable step LR scheduler.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        metavar="NUM_EPOCHS",
        type=int,
        default=500,
        help="Number of epochs",
    )
    parser.add_argument(
        "--warmup",
        metavar="WARMUP_EPOCHS",
        type=int,
        default=500,
        help="Number of warmup epochs with initial lr",
    )
    parser.add_argument(
        "--epoch-steps",
        metavar="STEP_EPOCHS",
        type=int,
        default=50,
        help="Number of epochs after which loss curve is updated and images are sampled",
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
        "--nf", type=int, default=32, metavar="NF", help="no. of filters (default: 32)"
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
        help="Use block checkpointing. Slow but uses less memory.",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Upsample type. Currently not supported.",
    )
    return parser, parser.parse_args()
