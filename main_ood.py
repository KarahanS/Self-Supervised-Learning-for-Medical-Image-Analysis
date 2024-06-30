# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os

from torch.utils import data
import hydra
import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import Mixup
from src.utils.enums import SplitType
from src.data.loader.medmnist_loader import MedMNISTLoader
from src.utils.setup import get_device
from src.utils.eval import get_representations
from src.utils.metrics import get_auroc_metric
from src.args.linear import parse_cfg
from src.ssl.methods.base import BaseMethod
from src.ssl.methods.linear import LinearModel
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import make_contiguous
from src.utils.enums import SplitType
import matplotlib as plt
import copy
import time
import numpy as np

try:
    from src.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True @ hydra.main(version_base="1.2")


def load_data(dataset, image_size, batch_size, num_workers, root):

    loader = MedMNISTLoader(
        data_flag=dataset,
        download=True,
        batch_size=batch_size,
        size=image_size,
        num_workers=num_workers,
        root=root,
    )

    train_dataclass = loader.get_data(SplitType.TRAIN)
    val_dataclass = loader.get_data(SplitType.VALIDATION)

    return loader, train_dataclass, val_dataclass


def build_data_loaders(dataset, image_size, batch_size, num_workers, root):

    # get train loaders
    logging.info("Preparing data loaders...")

    loader = MedMNISTLoader(
        data_flag=dataset,
        download=True,
        batch_size=batch_size,
        size=image_size,
        num_workers=num_workers,
        root=root,
    )

    train_dataclass = loader.get_data(SplitType.TRAIN, root=root)
    val_dataclass = loader.get_data(SplitType.VALIDATION, root=root)
    test_dataclass = loader.get_data(
        SplitType.TEST,
        root=root,
    )  # to be used afterwards for testing

    return loader, train_dataclass, val_dataclass, test_dataclass


def mahalanobis_distance(feat, mean, cov):
    diff = feat - mean
    inv_cov = torch.inverse(cov)
    mahalanobis_dist = torch.sqrt(torch.matmul(diff, torch.matmul(inv_cov, diff)))
    # diff : (feature_dim, )
    # inv_cov : (feature_dim, feature_dim)
    # mahalanobis_dist : (1, )
    return mahalanobis_dist


def get_mahalanobis_distances(feats, class_means, class_covs, unique_classes):
    mahalanobis_distances = []
    for feat in feats:
        min_dist = float("inf")
        for i, c in enumerate(unique_classes):
            dist = mahalanobis_distance(feat, class_means[i], class_covs[i])
            if dist < min_dist:
                min_dist = dist
        mahalanobis_distances.append((min_dist, c))
    return mahalanobis_distances


def get_samples_with_ratio(feats, labels, ratio):
    unique_classes = np.unique(labels)
    sampled_feats = []
    for c in unique_classes:
        c_feats = feats[labels == c]
        nsamples = len(c_feats) * ratio
        ood_class_feats = c_feats[:nsamples]
        sampled_feats.append(ood_class_feats)
    return torch.cat(sampled_feats, dim=0)


def get_feats(pretrained_model, train_dataclass, val_dataclass, device):
    train_feats = get_representations(
        pretrained_model, train_dataclass, device
    )  # returns the labels as well
    val_feats = get_representations(pretrained_model, val_dataclass, device)
    # combine features

    all_feats = torch.cat(
        [train_feats[0], val_feats[0]], dim=0
    )  # tensor  (# of samples, feature dim)
    all_labels = torch.cat(
        [train_feats[1], val_feats[1]], dim=0
    )  # tensor (# of samples, )
    return all_feats, all_labels


def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)

    if "vit" not in cfg.backbone.name:
        cfg.backbone.kwargs.pop("img_size", None)
        cfg.backbone.kwargs.pop("pretrained", None)

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            backbone.maxpool = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert (
        ckpt_path.endswith(".ckpt")
        or ckpt_path.endswith(".pth")
        or ckpt_path.endswith(".pt")
    )

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            logging.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]

    backbone.load_state_dict(state, strict=False)
    logging.info(f"Loaded {ckpt_path}")

    loader, train_dataclass, val_dataclass, test_dataclass = build_data_loaders(
        cfg.data.dataset,
        image_size=cfg.data.image_size,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        root=cfg.data.root,
    )

    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=loader.get_num_classes(),
        )
    device = get_device()

    id_feats, id_labels = get_feats(backbone, train_dataclass, val_dataclass, device)
    # get unique labels (assuming that training set includes all the unique labels)
    unique_classes = np.unique(train_dataclass.labels)

    # get mean and covariance for each class
    class_means = []
    class_covs = []

    for c in unique_classes:
        class_feats = id_feats[id_labels == c]  # torch.Tensor
        class_means.append(class_feats.mean(dim=0))
        class_covs.append(class_feats.T.cov())

    _, ood_train_dataclass, ood_val_dataclass = load_data(
        cfg.ood.data,
        cfg.data.image_size,
        cfg.optimizer.batch_size,
        cfg.data.num_workers,
        cfg.data.root,
    )

    ood_feats, ood_labels = get_feats(
        backbone, ood_train_dataclass, ood_val_dataclass, device
    )

    print("---- id_feats ----")
    id_sampled_feats = get_samples_with_ratio(id_feats, id_labels, cfg.ratio)
    print("---- ood_feats ----")
    ood_sampled_feats = get_samples_with_ratio(ood_feats, ood_labels, cfg.ratio)

    # get the mahalanobis distance for each sample:
    # the idea is to calcualte mahalonobis distance to each class and take the minimum distance

    print("---- id_mahalanobis_distances ----")
    id_mahalanobis_distances = get_mahalanobis_distances(
        id_sampled_feats, class_means, class_covs, unique_classes
    )
    print("---- ood_mahalanobis_distances ----")
    ood_mahalanobis_distances = get_mahalanobis_distances(
        ood_sampled_feats, class_means, class_covs, unique_classes
    )

    # plot the histogram of mahalanobis distances
    # maha_dist[0] = (maha_dist, closest class label)

    plt.hist([i[0] for i in id_mahalanobis_distances], bins=100, alpha=0.5, label="id")
    plt.hist(
        [i[0] for i in ood_mahalanobis_distances], bins=100, alpha=0.5, label="ood"
    )
    plt.legend()
    plt.show()
