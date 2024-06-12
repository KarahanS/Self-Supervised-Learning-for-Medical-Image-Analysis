# utils for evaluation

import torch
from torchmetrics import AUROC
from torchmetrics.classification import MultilabelAUROC
from torch.utils import data
from src.utils.enums import MedMNISTCategory


# make sure that the encoder doesn't have projection head attached to it
def get_representations(
    encoder, dataset, device, batch_size=64, sort=True, num_workers=7
):
    """
    Given a network encoder, pass the dataset through the encoder, remove the FC
    layers and return the encoded features.

    Args:
        network (torch.nn.Module): The network used for encoding features. (No projection head)
        dataset (torch.utils.data.Dataset): The input dataset.
        device (torch.device): Device used for computation.
        batch_size (int, optional): The batch size. Defaults to 64.
        sort (bool, optional): Sort the features by labels. Defaults to True.

    Returns:
        tuple: Tuple containing the encoded
            features and labels.
    """

    # Set network to evaluation mode
    encoder.eval()
    # Move network to specified device
    encoder.to(device)

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    feats, labels = [], []

    for batch_imgs, batch_labels in data_loader:
        # Move images to specified device
        batch_imgs = batch_imgs.to(device)
        # f(.)
        batch_feats = encoder(batch_imgs)

        # Detach tensor from current graph and move to CPU
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Remove extra axis
    labels = labels.squeeze()
    return (feats, labels)
