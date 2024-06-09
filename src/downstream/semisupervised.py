# TODO: Take a dataflag and a self-supervised pretrained model.
# Train the model a little bit further on the %1 or %10 of the labeled data.
# Save the checkpoint, then we will use this checkpoint to evaluate on the downstream tasks.

########## Task ##########:
# Load a self-supervised pretrained model
# Sample sample 1% or 10% of the labeled data in a class-balanced way
# Attach a linear layer with n_classes to the encoder and train it on this data using labels
# Save the model checkpoint
# Evaluate the model on downstream tasks
###########

from src.data.loader.medmnist_loader import MedMNISTLoader
from src.utils.enums import DatasetEnum
from src.utils.enums import SplitType
import torch.utils.data as data

from src.ssl.simclr.simclr import SimCLR

import torch
from pytorch_lightning import LightningModule


def load_pretrained_model(pretrained_path, cls: LightningModule):
    # Load the pretrained model
    pretrained_model = cls.load_from_checkpoint(pretrained_path, strict=False)
    return pretrained_model


# training the model on the sampled labeled data
def sample_labeled_data(cfg):
    #dataset, data_flag, augmentation_seq, batch_size, size, num_workers, percent=0.01):
    # Sample 1% of the labeled data
    # dataclass is a MedMNIST dataclass
    # percent is the percentage of the labeled data to sample
    # return a new dataclass with the sampled data

    train_params = cfg.Training.params
    downstream_params = cfg.Training.Downstream.params

    if cfg.Dataset.name == DatasetEnum.MEDMNIST:
        loader = MedMNISTLoader(
            data_flag=cfg.Dataset.params.medmnist_flag,
            augmentation_seq=cfg.Training.Downstream.augmentations,
            download=cfg.Dataset.params.download,
            batch_size=train_params.batch_size,
            size=cfg.Dataset.params.image_size,
            num_workers=cfg.Device.num_workers,
        )
    else:
        raise ValueError("Dataset not supported yet. Please use MedMNIST.")

    dataclass = loader.get_data(SplitType.TRAIN)

    class_indices = []
    num_classes = loader.get_num_classes()

    n_samples = int(downstream_params.percent * len(dataclass))
    samples_per_class = n_samples // num_classes

    labels = torch.cat(
        [batch_labels for _, batch_labels in loader.load(dataclass, shuffle=True)]
    ).flatten()

    for idx in range(num_classes):
        samples = torch.where(labels == idx)[0]
        perm = torch.randperm(len(samples))
        # sample "samples_per_class" samples from each class
        # if there are less than "samples_per_class" samples in a class, sample all of them
        samples = samples[perm[:samples_per_class]]
        class_indices.append(samples)

    indices = torch.cat(class_indices)  # convert list of tensors to a single tensor
    return data.Subset(dataclass, indices)
