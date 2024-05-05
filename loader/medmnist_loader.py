import os
import numpy as np
import medmnist
from medmnist import INFO, Evaluator

import torch.utils.data as data
import torchvision.transforms as transforms
from utils.constants import MEDMNIST_DATA_DIR

## TODO: Test data will be used for the downstream task evaluation.
## Use training and validation data for self-supervised learning.


class MedMNISTLoader:
    def __init__(self, data_flag, download, batch_size, size=28, views=2):
        self.info = INFO[data_flag.__str__()]
        DataClass = getattr(medmnist, self.info["python_class"])

        # TODO: self.transforms

        try:
            self.train_data = DataClass(
                split="train",
                transform=ContrastiveTransformations(self.transforms, views),
                download=download,
                root=MEDMNIST_DATA_DIR,
                size=size,
            )
            self.test_data = DataClass(
                split="test",
                transform=ContrastiveTransformations(self.transforms, views),
                download=download,
                root=MEDMNIST_DATA_DIR,
                size=size,
            )
        except:
            print(
                f"Download failed. Please make sure that the dataset folder {MEDMNIST_DATA_DIR} exists."
            )
            return
        self.train_loader = data.DataLoader(
            dataset=self.train_data, batch_size=batch_size, shuffle=True
        )
        self.train_loader_at_eval = data.DataLoader(
            dataset=self.train_data, batch_size=2 * batch_size, shuffle=False
        )
        self.test_loader = data.DataLoader(
            dataset=self.test_data, batch_size=2 * batch_size, shuffle=False
        )

        print(self.train_data)
        print("===================")
        print(self.test_data)

    def get_loaders(self):
        return self.train_loader, self.train_loader_at_eval, self.test_loader

    def show_info(self):
        print(f"Task: {self.info['task']}")
        print(f"Number of channels: {self.info['n_channels']}")
        print(f"Number of labels: {len(self.info['label'])}")
        print(f"Meaning of labels: {self.info['label']}")

    def display_data(self):
        self.train_data.montage(length=1).show()


# apply given transformations to the current image n_view times
class ContrastiveTransformations:
    def __init__(self, contrastive_transformations, n_views=2):
        self.contrastive_transformations = contrastive_transformations
        self.n_views = n_views

    def __call__(self, x):
        return [self.contrastive_transformations(x) for _ in range(self.n_views)]
