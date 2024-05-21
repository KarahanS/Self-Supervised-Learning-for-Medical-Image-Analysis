import argparse

import src.utils.setup as setup
from src.utils.enums import DatasetEnum, MedMNISTCategory, SSLMethod, DownstreamMethod
from src.utils.augmentations import AugmentationSequenceType

from src.utils.config.config import Config
# import your train with the name of the approach
from src.ssl.simclr.train import train as simclr_train
from src.downstream.eval.train import train as eval_train

parser = argparse.ArgumentParser(
    description="PyTorch SSL Training for Medical Image Analysis"
)

parser.add_argument(
    "--cfg-path", default=None, help="Path to the configuration file."
)


def set_augmentation(config):
    """
    If augmentation is not specified, set the default augmentation sequence to "default" for self-supervised learning.
    For downstream tasks, set the augmentation sequence to "preprocess".
    """

    if "Pretrain" in config.Training:
        if not config.Training.Pretrain.augmentations:  # Empty list
            config.Training.Pretrain.augmentations = AugmentationSequenceType.DEFAULT
    else:
        config.Training.Downstream.augmentations = AugmentationSequenceType.PREPROCESS


def main():
    args = parser.parse_args()

    cfg = Config(args.cfg_path)
    args = cfg.get_config()

    setup.configure_paths(cfg)

    # check if gpu training is available
    setup.setup_device(cfg)

    if cfg.Dataset.name == DatasetEnum.MIMETA:
        cfg.Dataset.params.image_size = 224  # there is no other option for MIMETA dataset

    set_augmentation(cfg)

    # main can be used either for self-supervised pretraining or downstream task evaluation
    if "Pretrain" in cfg.Training:
        assert (
            cfg.Training.Pretrain.params.n_views == 2
        ), "Only two view training is supported. Please use --n-views 2."

        # Dataset should be read in the train.py of related SSL method
        if cfg.Training.Pretrain == SSLMethod.SIMCLR:
            simclr_train(cfg)
        else:
            raise ValueError("Other SSL methods are not supported yet.")
    else:  # Downstream
        if args.eval_method in [DownstreamMethod.LINEAR, DownstreamMethod.NONLINEAR]:
            eval_train(cfg)  # logistic regression or mLP
        else:
            raise ValueError("Other evaluation methods are not supported yet.")

if __name__ == "__main__":
    main()
