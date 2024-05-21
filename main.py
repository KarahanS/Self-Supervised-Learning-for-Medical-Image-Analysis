import argparse

import src.utils.setup as setup
from src.utils.enums import DatasetEnum
from src.utils.enums import MedMNISTCategory
from src.utils.augmentations import AugmentationSequenceType

# import your train with the name of the approach
from src.ssl.simclr.train import train as simclr_train
from src.downstream.eval.train import train as eval_train

parser = argparse.ArgumentParser(
    description="PyTorch SSL Training for Medical Image Analysis"
)
parser.add_argument(
    "-data", metavar="DIR", default="./datasets", help="path to dataset"
)
# hidden dimension can be used for either projection head (pretraining) or MLP (downstream)
parser.add_argument(
    "-hd", "--hidden-dim", default=512, type=int, help="hidden dimension"
)
parser.add_argument(
    "-pt",
    "--pretrained",
    action=argparse.BooleanOptionalAction,
    help="Use a supervised-pretrained model for further self-supervised pretraining, or pre-train a new model from scratch.",
)
parser.add_argument("--output-dim", default=128, type=int, help="output dimension")
parser.add_argument(  # datasets can be MedMNIST or MIMeta
    "-dataset_name",
    default="medmnist",
    help="dataset name can be either medmnist or mimeta. Default: medmnist.",
    type=DatasetEnum,
    choices=list(
        DatasetEnum
    ),  # careful! comparing dataset_name with a string won't work, compare it with enum.
)

parser.add_argument(
    "--data-flag",
    default=MedMNISTCategory.PATH,
    type=MedMNISTCategory,
    choices=list(MedMNISTCategory),
    help="data flag for MedMNIST dataset",
)
parser.add_argument(
    "--size",
    default=28,
    type=int,
    metavar="N",
    help="image size (default: 28)",
    choices=[28, 64, 128, 224],
)
parser.add_argument(
    "-e",
    "--encoder",
    metavar="ENC",
    default="resnet50",
    choices=model_names,  # TODO: Modify here if you use different models as backbone
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-nw",
    "--num_workers",
    default=7,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 7)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-aug",
    "--augmentation",
    # default=AugmentationSequenceType.DEFAULT,
    type=AugmentationSequenceType,
    help="Augmentation sequence to use. Check utils.augmentations for details. Use 'preprocess' for downstream tasks.",
    choices=list(AugmentationSequenceType),
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    default=0.0003,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "-wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("--seed", default=42, type=int, help="seed for reproducibility. ")
parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training.",
)

parser.add_argument(
    "--out_dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--log-every-n-steps", default=100, type=int, help="Log every n steps"
)
parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)",
)
parser.add_argument(
    "--n-views",
    default=2,
    type=int,
    metavar="N",
    help="Number of views for contrastive learning training.",
)
parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")
parser.add_argument("--ssl-method", default="simclr", help="SSL method to use.")
parser.add_argument(
    "--log",
    default="wandb",
    choices=["wandb", "tb", "off"],
    help="Specify the logging tool to use: 'wandb', 'tensorboard', or 'off' to disable logging. Defaults to Wandb.",
)
parser.add_argument(
    "--action",
    default="pretrain",
    choices=["pretrain", "downstream"],
    help="Use 'pretrain' for self-supervised pretraining and 'downstream' for running a trained model on a downstream task",
)
parser.add_argument(
    "--eval-method",
    default="linear",
    choices=["linear", "nonlinear", "semi-supervised"],
    help="Use which evaluation method to use to measure the quality of the learned representations.",
)
parser.add_argument(
    "--pretrained-path",
    default=None,
    help="Path to the pretrained model to use for downstream tasks.",
)


def set_augmentation(args):
    """
    If augmentation is not specified, set the default augmentation sequence to "default" for self-supervised learning.
    For downstream tasks, set the augmentation sequence to "preprocess".
    """
    if args.action == "pretrain":
        if not args.augmentation:
            args.augmentation = AugmentationSequenceType.DEFAULT
    else:
        args.augmentation = AugmentationSequenceType.PREPROCESS


def main():
    args = parser.parse_args()
    set_augmentation(args)
    print(args)
    if args.dataset_name == DatasetEnum.MIMETA:
        args.size = 224  # there is no other option for MIMETA dataset

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    setup.setup_device(args.seed)

    # main can be used either for self-supervised pretraining or downstream task evaluation
    if args.action == "pretrain":
        # Dataset should be read in the train.py of related SSL method
        if args.ssl_method == "simclr":
            simclr_train(**vars(args))
        else:
            raise ValueError("Other SSL methods are not supported yet.")
    else:
        if args.eval_method in ["linear", "nonlinear"]:
            eval_train(**vars(args))  # logistic regression or mLP
        else:
            raise ValueError("Other evaluation methods are not supported yet.")


if __name__ == "__main__":
    main()
