import argparse

import src.utils.setup as setup
from src.utils.enums import DatasetEnum
from src.utils.enums import MedMNISTCategory
from src.utils.augmentations import AugmentationSequenceType

from torchvision import models
import torchvision.models as models

from src.ssl.simclr.train import train as simclr_train


model_names = models.list_models()

parser = argparse.ArgumentParser(
    description="PyTorch SSL Training for Medical Image Analysis"
)
parser.add_argument(
    "-data", metavar="DIR", default="./datasets", help="path to dataset"
)
parser.add_argument(
    "-hd", "--hidden-dim", default=512, type=int, help="hidden dimension"
)
parser.add_argument(
    "-pt", "--pretrained", default=True, action=argparse.BooleanOptionalAction
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
    "--data_flag",
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
    default=AugmentationSequenceType.DEFAULT,
    type=AugmentationSequenceType,
    help="Augmentation sequence to use. Check utils.augmentations for details.",
    choices=list(AugmentationSequenceType),
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0003,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
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
parser.add_argument("--max-epochs", default=100, type=int, help="Number of epochs.")


def main():
    args = parser.parse_args()
    if args.dataset_name == DatasetEnum.MIMETA:
        args.size = 224  # there is no other option for MIMETA dataset

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    device = setup.setup_device(args.seed)

    # Dataset should be read in the train.py of related SSL method
    if args.ssl_method == "simclr":
        simclr_train(**vars(args))
    else:
        raise ValueError("Other SSL methods are not supported yet.")


if __name__ == "__main__":
    main()
