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
from src.utils.enums import MedMNISTCategory
from src.utils.enums import SplitType
import torch.utils.data as data
from torch.utils.data import Subset
import numpy as np

import torch
from pytorch_lightning import LightningModule


def load_pretrained_model(pretrained_path, cls: LightningModule):
    # Load the pretrained model
    pretrained_model = cls.load_from_checkpoint(pretrained_path, strict=False)
    return pretrained_model

def sample_balanced_data(dataset, train_fraction):
    """
    Sample balanced data from the dataset. Each class will have the same proportion as in the original dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset object
        train_fraction (float): Fraction of the data to sample
        
    Returns:
        Subset: A Subset of the original dataset with balanced classes
    """
    # Check if train_fraction is within a valid range
    if not (0 < train_fraction <= 1):
        raise ValueError("train_fraction must be between 0 and 1")
    
    # Get the labels from the dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate the number of samples to be drawn
    total_samples = int(len(dataset) * train_fraction)
    
    # Get the class distribution in the original dataset
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    class_distribution = class_counts / len(labels)
    
    # Sample data based on the class distribution
    sampled_indices = []
    for class_label, proportion in zip(unique_classes, class_distribution):
        # Number of samples for this class
        n_samples = int(total_samples * proportion)
        
        # Get all indices for this class
        class_indices = np.where(labels == class_label)[0]
        
        # Ensure that we do not sample more than what is available without replacement
        if n_samples > len(class_indices):
            raise ValueError(f"Requested more samples ({n_samples}) than available for class {class_label}. Reduce the train_fraction.")
        
        # Sample without replacement
        sampled_class_indices = np.random.choice(class_indices, n_samples, replace=False)
        
        # Append the sampled indices
        sampled_indices.extend(sampled_class_indices)
    
    # Create a Subset with the sampled indices
    sampled_subset = Subset(dataset, sampled_indices)
    
    # Validate that the class distribution is correct
    # is_valid = validate_class_distribution(dataset, sampled_subset)
    # print("Validation result:", is_valid)
    return sampled_subset


# training the model on the sampled labeled data
def sample_labeled_data(dataset_name, download, batch_size, image_size, num_workers, train_fraction, root):
    #dataset, data_flag, augmentation_seq, batch_size, size, num_workers, percent=0.01):
    # Sample 1% of the labeled data
    # dataclass is a MedMNIST dataclass
    # percent is the percentage of the labeled data to sample
    # return a new dataclass with the sampled data

    if dataset_name in MedMNISTCategory._value2member_map_:
        loader = MedMNISTLoader(
            data_flag=dataset_name,
            download=download,
            batch_size=batch_size,
            size=image_size,
            num_workers=num_workers,
            root=root,
        )
    else:
        raise ValueError("Dataset not supported yet. Please use MedMNIST.")

    dataclass = loader.get_data(SplitType.TRAIN, root=root)

    class_indices = []
    num_classes = loader.get_num_classes()
    n_samples = int(train_fraction * len(dataclass))
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

def validate_class_distribution(dataset, sampled_subset):
    """
    Validate that the percentage of each class in the sampled subset is the same as in the original dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): Original dataset object
        sampled_subset (Subset): Sampled subset of the dataset
        
    Returns:
        bool: True if distributions match, False otherwise
    """
    # Get the labels from the original dataset
    original_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Get the labels from the sampled subset
    sampled_labels = np.array([dataset[i][1] for i in sampled_subset.indices])
    
    # Calculate the class distribution in the original dataset
    original_unique_classes, original_class_counts = np.unique(original_labels, return_counts=True)
    original_class_distribution = original_class_counts / len(original_labels)
    
    # Calculate the class distribution in the sampled subset
    sampled_unique_classes, sampled_class_counts = np.unique(sampled_labels, return_counts=True)
    sampled_class_distribution = sampled_class_counts / len(sampled_labels)
    
    # Validate that the distributions match
    for class_label in original_unique_classes:
        original_proportion = original_class_distribution[original_unique_classes == class_label][0]
        sampled_proportion = sampled_class_distribution[sampled_unique_classes == class_label][0]
        if not np.isclose(original_proportion, sampled_proportion, atol=1e-2):  # Allowing a small tolerance
            print(f"Class {class_label} proportion mismatch: original={original_proportion}, sampled={sampled_proportion}")
            return False
    
    return True

# Example validation:
# 

