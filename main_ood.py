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
import random
from typing import List

from torch.utils import data
import hydra
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lightning.pytorch import Trainer, seed_everything
from src.downstream.semisupervised import sample_balanced_data
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import Mixup
from src.utils.enums import SplitType
from src.data.loader.medmnist_loader import MedMNISTLoader
from torch.utils.data import Subset
from src.utils.setup import get_device
from src.utils.eval import get_representations
from src.utils.metrics import get_auroc_metric, get_balanced_accuracy_metric
from src.args.linear import parse_cfg
from src.ssl.methods.base import BaseMethod
from src.ssl.methods.linear import LinearModel
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import make_contiguous
from src.utils.enums import SplitType
from src.utils.mahalanobis import calculate_gaussian_parameters, calculate_confidence_scores, generate_ood_labels, calculate_scores
import copy
import time

try:
    from src.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True


def build_data_loaders(dataset, image_size, batch_size, num_workers, root, train_fraction=1.0):

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
    test_dataclass = loader.get_data(SplitType.TEST, root=root)  # to be used afterwards for testing

    # # adjust the size of the train dataset in accordance with the train_fraction of the original size
    if train_fraction < 1.0:
        train_dataclass = sample_balanced_data(train_dataclass, train_fraction)
        logging.info(f"Training on {len(train_dataclass)} samples on fraction {train_fraction}")

    return loader, train_dataclass, val_dataclass, test_dataclass


def generate_seeds(n: int) -> List[int]:
    random.seed(time.time())
    return [random.randint(0, 2 ** 32 - 1) for _ in range(n)]

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)

    if "vit" not in cfg.backbone.name:
        cfg.backbone.kwargs.pop('img_size',None)
        cfg.backbone.kwargs.pop('pretrained',None)
    
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
                "You are using an older checkpoint. Use a new one as some issues might arise."
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
        train_fraction=cfg.data.train_fraction,
    )

    # Build the data loaders for OOD dataset
    ood_loader, train_ood_dataclass, val_ood_dataclass, test_ood_dataclass = build_data_loaders(
        cfg.ood_data.dataset,
        image_size=cfg.ood_data.image_size,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.ood_data.num_workers,
        root=cfg.ood_data.root,
        train_fraction=cfg.ood_data.train_fraction,
    )

    device = get_device()
    num_classes = loader.get_num_classes()

    train_feats_tuple = get_representations(backbone, train_dataclass, device)
    val_feats_tuple = get_representations(backbone, val_dataclass, device)
    test_feats_tuple = get_representations(backbone, test_dataclass, device)

    if cfg.ood_data.dataset == cfg.data.dataset:
        train_ood_feats_tuple = train_feats_tuple
        val_ood_feats_tuple = val_feats_tuple
        test_ood_feats_tuple = test_feats_tuple
        ood_feats = torch.cat((train_ood_feats_tuple[0], val_ood_feats_tuple[0], test_ood_feats_tuple[0]), dim=0)

    else:
        train_ood_feats_tuple = get_representations(backbone, train_ood_dataclass, device)
        val_ood_feats_tuple = get_representations(backbone, val_ood_dataclass, device)
        test_ood_feats_tuple = get_representations(backbone, test_ood_dataclass, device)
        ood_feats = torch.cat((train_ood_feats_tuple[0], val_ood_feats_tuple[0], test_ood_feats_tuple[0]), dim=0)

    feature_dim = test_feats_tuple[0][0].shape[0]

    # Reshape the feats into C x N_c x D 
    train_feats_reshaped = [[] for _ in range(num_classes)]

    # Move each class' features into the first dimension
    for i in range(num_classes):
        matched = train_feats_tuple[1] == i
        train_feats_reshaped[i] = train_feats_tuple[0][matched]

    # Get OOD score
    feats1 = test_feats_tuple[0]
    feats2 = ood_feats

    means, covs = calculate_gaussian_parameters(train_feats_reshaped)
    confidence_scores = calculate_confidence_scores(feats1, means, covs)
    confidence_scores_ood = calculate_confidence_scores(feats2, means, covs)
    ood_labels = generate_ood_labels(feats1,feats2)

    # concat the in-distribution and out-of-distribution confidence scores
    conf_scores = torch.cat((confidence_scores, confidence_scores_ood))
    scores = calculate_scores(conf_scores, ood_labels)
 
    print(f'Auroc : {scores[0]}')
    print(f'Average precision scores : {scores[1]}')

    # get model_lr after parsing teh config : lr-0.01-wd-0.0001 
    if cfg.to_csv.enabled:
        csv_file = cfg.to_csv.name
        if not csv_file.endswith(".csv"):
            csv_file += ".csv"

        # Check if the CSV file exists
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, "a") as f:
            # If the file doesn't exist, write the header
            if not file_exists:
                f.write(
                    "model_name,classifier_name,id_dataset,ood_dataset,auroc,average_precision\n"
                )

            # Write the model data
            f.write(
                f"{cfg.name},{cfg.downstream_classifier.name},{cfg.data.dataset},{cfg.ood_data.dataset},{scores[0]},{scores[1]}\n"
            )


if __name__ == "__main__":
    main()