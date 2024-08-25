import os
import numpy as np
import medmnist
from medmnist import INFO

import torch.utils.data as data
import torchvision.transforms as transforms
from src.utils.enums import SplitType, MedMNISTCategory
import logging

MEDMNIST_DATASETS = [
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "pathmnist",
    "chestmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "breastmnist",
    "tissuemnist",
    "bloodmnist",
]


def to_rgb(img):
    return img.convert("RGB")


def get_data_class(data_flag):
    info = INFO[data_flag.__str__()]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass


def get_single_label(lbl):
    if len(lbl) == 1:
        return np.array(lbl[0])
    else:
        raise ValueError("Multiple labels detected. This is not implemented yet")


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
            data_flag: Data flag for MedMNIST dataset
            download: Whether to download the dataset if not present
            batch_size: Mini-batch size
            size: Image size
            views: Number of views for contrastive learning training

        """
        self.info = INFO[data_flag.__str__()]
        self.DataClass = getattr(medmnist, self.info["python_class"], size)

        # augmentation_seq = AugmentationSequenceType(augmentation_seq)
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

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

class CombinedMedMNIST(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._cumulative_sizes()
        self.label_offsets = self._label_offsets()
        self.num_classes = sum(len(dataset.info["label"]) for dataset in datasets)
    def _cumulative_sizes(self):
        cumulative_sizes = []
        total_size = 0
        for dataset in self.datasets:
            total_size += len(dataset)
            cumulative_sizes.append(total_size)
        return cumulative_sizes
    
    def _label_offsets(self):
        offsets = [0]
        for i, dataset in enumerate(self.datasets[:-1]):  # Exclude the last dataset
            max_label = np.max([label[0] for label in dataset.labels])
            offsets.append(offsets[-1] + max_label + 1)
        return offsets
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise IndexError("Index out of bounds")
        dataset_idx = next(i for i, size in enumerate(self.cumulative_sizes) if size > index)
        if dataset_idx == 0:
            dataset_index = index
        else:
            dataset_index = index - self.cumulative_sizes[dataset_idx - 1]

        # update the label to be unique across all datasets 
        img,label = self.datasets[dataset_idx][dataset_index]
        label += self.label_offsets[dataset_idx]

        return img, label
    