import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List

def calculate_gaussian_parameters(feats: List[torch.Tensor]):
    '''
    Calculate the mean and covariance of features for each class, then average over all samples.
    
    Args:
        feats (List[torch.Tensor]): Features of each class, of size (C, N_c, D), where C is the number of classes,
                                    N_c is the number of samples per class, and D is the feature dimension.
    
    Returns:
        means (torch.Tensor): Mean of the features of the size (C, D).
        covs (torch.Tensor): Shared covariance matrix of the features of the size (D, D), averaged over all samples.
    '''

    num_classes = len(feats)
    feature_dim = feats[0].shape[1]

    # Compute the means for each class
    means = torch.zeros((num_classes, feature_dim))
    for i in range(num_classes):
        means[i] = torch.mean(feats[i], axis=0)

    # Initialize covariance matrix and count total samples
    covs = torch.zeros((feature_dim, feature_dim))
    total_samples = sum([feats[i].shape[0] for i in range(num_classes)])  # Total number of samples across all classes

    # Compute covariance matrix (averaged over all samples)
    for i in range(num_classes):
        diff = feats[i] - means[i]  # Center the data for the class
        covs += diff.T @ diff  # Accumulate the covariance contributions without dividing by class size

    # Divide by total number of samples (minus one for unbiased covariance estimate)
    covs /= (total_samples - num_classes)  # Total samples minus number of classes for unbiased estimate

    # Add small value to diagonal for numerical stability
    covs += torch.eye(feature_dim) * 1e-6

    return means, covs


def calculate_confidence_scores(test_feats: torch.Tensor, means: torch.Tensor, covs: torch.Tensor):
    '''
    Calculate the Mahalanobis distance between test features and the class means using matrix multiplication.

    Args:
        test_feats (torch.Tensor): Features of the test set of size (N, D)
        means (torch.Tensor): Mean of the features of the size (C, D)
        covs (torch.Tensor): Covariance matrix of the features of the size (D, D)

    Returns:
        confidence_score (torch.Tensor): Confidence score of the features of the size (N) indicating the OOD score
    '''

    # Calculate the inverse covariance matrix
    covs_inv = torch.inverse(covs).float()  # Convert to float32 if necessary

    num_class = means.shape[0]
    num_samples = test_feats.shape[0]
    distances = torch.zeros((num_class, num_samples), dtype=torch.float32)  # Ensure float32

    # Mahalanobis distance calculation
    for i in range(num_class):
        diff = (test_feats - means[i]).float()  # Shape: (N, D)
        # Mahalanobis distance: (x - mean)^T * cov_inv * (x - mean)
        # First do (covs_inv @ diff^T), then diff @ result
        left = torch.matmul(diff, covs_inv)  # (N, D) @ (D, D) -> (N, D)
        distance = torch.sum(left * diff, axis=1)  # Element-wise multiplication and sum -> (N,)
        distances[i] = distance

    # Confidence score is the negative of the minimum Mahalanobis distance (highest similarity)
    confidence_score = torch.min(distances, axis=0).values
    return confidence_score


def generate_ood_labels(feats1,feats2):
    '''
    Generate 0-1 labels for in-distribution and out-of-distribution samples.
    Assume feats1 is in-distribution and feats2 is out-of-distribution.
    '''
    N1 = feats1.shape[0]
    N2 = feats2.shape[0]
    labels = torch.ones(N1+N2)
    labels[:N1] = 0
    return labels

def calculate_scores(preds, labels):
    '''
    Given ID-OOD predictions and labels, calculate the AUROC and AUPR scores.
    '''
    auroc = roc_auc_score(labels, preds)
    aupr = average_precision_score(labels, preds)
    return auroc, aupr
