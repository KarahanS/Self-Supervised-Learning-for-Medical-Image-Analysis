import os
import numpy as np
import medmnist
from medmnist import INFO, Evaluator

import torch.utils.data as data
import torchvision.transforms as transforms
import src.utils.constants as const
from src.utils.augmentations import get_augmentation_sequence
from src.utils.enums import SplitType, MedMNISTCategory
import logging

## TODO: Test data will be used for the downstream task evaluation.
## Use training and validation data for self-supervised learning.


class MedMNISTLoader:
    def __init__(
        self,
        data_flag: MedMNISTCategory,
        download,
        batch_size,
        num_workers,
        root,
        size=28,
        views=2,
    ):
        """
        Loader for MedMNIST dataset

        Args:
            augmentation_seq: Augmentation sequence to use.
            data_flag: Data flag for MedMNIST dataset
            download: Whether to download the dataset if not present
            batch_size: Mini-batch size
            size: Image size
            views: Number of views for contrastive learning training

        """
        self.info = INFO[data_flag.__str__()]
        self.DataClass = getattr(medmnist, self.info["python_class"], size)

        try:
            # augmentation_seq = AugmentationSequenceType(augmentation_seq)
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
            )
        except KeyError:
            raise ValueError("Augmentation flag is invalid")

        self.data_flag = data_flag
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.views = views
        self.root = root
        self.num_classes = len(self.info["label"])

    # call with contrastive = True for SSL, call with contrastive = False for downstream tasks
    def get_data_and_load(self, split, shuffle, contrastive=False):
        # TODO: Add support for class-balanced loading # https://github.com/j-freddy/simclr-medical-imaging/blob/main/downloader.py#L31
        """
        Creates the dataclass and returns a dataloader according to the split.
        """

        transform = (
            ContrastiveTransformations(self.transforms, self.views)
            if contrastive
            else self.transforms
        )

        dataclass = self.get_data(split, self.root, transform)
        logging.info(dataclass)
        logging.info("===================")
        return data.DataLoader(
            dataset=dataclass,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    # overload
    def load(self, dataclass, shuffle):
        return data.DataLoader(
            dataset=dataclass,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_data(self, split, root, transform=None):
        if transform is None:
            transform = self.transforms
        dataclass = self.DataClass(
            split=split.value,
            transform=transform,
            download=self.download,
            root=root,
            size=self.size,
        )
        return dataclass

    def show_info(self):
        logging.info(f"Task: {self.info['task']}")
        logging.info(f"Number of channels: {self.info['n_channels']}")
        logging.info(f"Number of labels: {len(self.info['label'])}")
        logging.info(f"Meaning of labels: {self.info['label']}")

    def display_data(self):
        self.train_data.montage(length=1).show()

    def get_num_classes(self):
        return self.num_classes


# apply given transformations to the current image n_view times
class ContrastiveTransformations:
    def __init__(self, contrastive_transformations, n_views=2):
        self.contrastive_transformations = contrastive_transformations
        self.n_views = n_views

    def __call__(self, x):
        return [self.contrastive_transformations(x) for _ in range(self.n_views)]
