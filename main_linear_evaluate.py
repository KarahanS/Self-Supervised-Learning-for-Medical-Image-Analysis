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
from src.utils.metrics import get_auroc_metric
from src.args.linear import parse_cfg
from src.ssl.methods.base import BaseMethod
from src.ssl.methods.linear import LinearModel
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import make_contiguous
from src.utils.enums import SplitType
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

    state = torch.load(cfg.checkpoint_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            logging.warn(
                "You are using an older checkpoint. Use a new one as some issues might arise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]  # Deletes also superfluous weights for backbone (e.g. classifier weights)

    backbone.load_state_dict(state, strict=False)
    logging.info(f"Loaded {cfg.checkpoint_path}")

    loader, train_dataclass, val_dataclass, test_dataclass = build_data_loaders(
        cfg.data.dataset,
        image_size=cfg.data.image_size,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        root=cfg.data.root,
        train_fraction=cfg.data.train_fraction,
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

    print("Training on", len(train_dataclass), "samples on fraction", cfg.data.train_fraction)

    if not cfg.finetune:
        train_feats_tuple = get_representations(backbone, train_dataclass, device)
        test_feats_tuple = get_representations(backbone, test_dataclass, device)

        test_feats = data.TensorDataset(test_feats_tuple[0], test_feats_tuple[1])

        feature_dim = feature_dim = train_feats_tuple[0][0].shape[0]
    else:
        test_feats = test_dataclass

        # Get the feature dimensions from the backbone's output
        # TODO: Look for a more elegant way than passing a temporary input
        feature_dim = backbone(torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)).shape[1]

    num_classes = loader.get_num_classes()

    best_model = None

    # add tqdm here
    if isinstance(cfg.optimizer.lr, int) or isinstance(cfg.optimizer.lr, float):
        cfg.optimizer.lr = [cfg.optimizer.lr]
    if isinstance(cfg.optimizer.weight_decay, int) or isinstance(cfg.optimizer.weight_decay, float):
        cfg.optimizer.weight_decay = [cfg.optimizer.weight_decay]
    
    lr = cfg.optimizer.lr[0]
    wd = cfg.optimizer.weight_decay[0]
    best_comb = (lr, wd)

    lr, wd = best_comb
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = wd
    # now we have the best_model
    model = LinearModel(
        backbone,
        mixup_func=mixup_func,
        cfg=cfg,
        num_classes=num_classes,
        feature_dim=feature_dim,  # give feature dimensions
    )

    callbacks = []
    wandb_run_id = None
    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs
    }
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": (
                DDPStrategy(find_unused_parameters=False)
                if cfg.strategy == "ddp"
                else cfg.strategy
            ),
        }
    )
    trainer = Trainer(**trainer_kwargs)
    test_loader = loader.load(test_feats, shuffle=False)

    best_model = LinearModel.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        backbone=backbone,
        cfg=cfg,
        num_classes=num_classes,
        feature_dim=feature_dim,
    )
    
    test_result = trainer.test(best_model, dataloaders=test_loader, verbose=False)
    test_acc = test_result[0]["test_acc"]

    test_auroc = get_auroc_metric(
        best_model, test_loader, loader.get_num_classes(), cfg.data.task
    )

    if cfg.wandb.enabled:
        wandb_logger.log_metrics({"auroc": test_auroc})
    logging.info(test_auroc)
    wandb_logger.log_metrics({"weight decay": wd})

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
                    "model_name,downstream_classifier_name,dataset,learning_rate,weight_decay,test_acc,test_auroc\n"
                )

            # Write the model data
            f.write(
                f"{cfg.name},{cfg.downstream_classifier.name},{cfg.data.dataset},{lr},{wd},{test_acc},{test_auroc}\n"
            )


if __name__ == "__main__":
    main()