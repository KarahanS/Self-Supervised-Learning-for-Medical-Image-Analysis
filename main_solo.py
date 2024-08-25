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
import os
import itertools
import hydra
import torch
import logging
import re

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf, listconfig
from src.args.pretrain import parse_cfg,_N_CLASSES_MEDMNIST
from src.data.classification_dataloader import (
    prepare_data as prepare_data_classification,
)
from src.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from src.ssl.methods import METHODS
from src.ssl.methods.linear import LinearModel
from src.data.loader.medmnist_loader import MedMNISTLoader,MEDMNIST_DATASETS
from src.utils.enums import SplitType
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.eval import get_representations
from src.utils.misc import make_contiguous, omegaconf_select
from torch.utils import data

try:
    from src.data.dali_dataloader import (
        PretrainDALIDataModule,
        build_transform_pipeline_dali,
    )
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


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

def train_linear_head(cfg : DictConfig, backbone, loader, train_dataclass, val_dataclass, **trainer_kwargs):
    # freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    linear_cfg = cfg.copy()

    # For now, we are using a fixed linear model - manually set them for nop
    linear_cfg.optimizer = linear_cfg.grid_search.optimizer
    linear_cfg.downstream_classifier = linear_cfg.grid_search.downstream_classifier
    linear_cfg.max_epochs = linear_cfg.grid_search.pretrain_max_epochs
    linear_cfg.scheduler = linear_cfg.grid_search.scheduler

    OmegaConf.update(linear_cfg, "data.task", 'multiclass')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.data.dataset in MEDMNIST_DATASETS:
        train_feats_tuple = get_representations(backbone, train_dataclass, device)
        val_feats_tuple = get_representations(backbone, val_dataclass, device)

        train_feats = data.TensorDataset(train_feats_tuple[0], train_feats_tuple[1])
        val_feats = data.TensorDataset(val_feats_tuple[0], val_feats_tuple[1])
        
        feature_dim = feature_dim = train_feats_tuple[0][0].shape[0]
        num_classes = _N_CLASSES_MEDMNIST[linear_cfg.data.dataset] 

        mixup_func = None

        linear_model = LinearModel(
            backbone=backbone,
            cfg=linear_cfg,
            num_classes=num_classes,
            mixup_func=mixup_func,
            feature_dim=feature_dim,  # give feature dimensions
        )

        linear_train_loader = loader.load(train_feats, shuffle=True)
        linear_validation_loader = loader.load(val_feats, shuffle=False)
    else:
        linear_train_loader, linear_validation_loader = None, None # placeholder
        raise NotImplementedError("For now, only MedMNIST datasets are supported for grid search")

    # Refresh it as DDPStrategy should be reinitialized
    linear_trainer_kwargs = trainer_kwargs.copy()
    linear_trainer_kwargs.update(
        {
            "enable_checkpointing": False,
            "strategy": (
                DDPStrategy(find_unused_parameters=False)
                if cfg.strategy == "ddp"
                else cfg.strategy
            ),
            "max_epochs": linear_cfg.max_epochs,
        }
    )

    linear_trainer = Trainer(**linear_trainer_kwargs)
    linear_trainer.fit(linear_model, linear_train_loader, linear_validation_loader)
    out = linear_trainer.validate(linear_model, linear_validation_loader)

    # Get the validation accuracy
    return linear_trainer, linear_model, out
    
def train_ssl_model(cfg : DictConfig, grid_search_enabled: bool = False):
    cfg = parse_cfg(cfg)
    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae"]

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (
        cfg.data.no_labels or cfg.data.val_path is None
    ):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format

        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    # pretrain dataloader
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        )
        callbacks.append(ckpt)

    if omegaconf_select(cfg, "auto_umap.enabled", False):
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
            # save_dir=os.path.join(cfg.checkpoint.dir, "wandb"),
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

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if grid_search_enabled:
        if cfg.data.dataset in MEDMNIST_DATASETS:
            loader, train_dataclass, val_dataclass, _ = build_data_loaders(cfg.data.dataset,
                                                                                        64,
                                                                                        cfg.optimizer.batch_size,
                                                                                        8,
                                                                                        cfg.data.train_path)
        else:
            linear_train_loader, linear_validation_loader = None, None # placeholder
            raise NotImplementedError("For now, only MedMNIST datasets are supported for grid search")

        # Train a linear head
        linear_trainer, linear_model, out = train_linear_head(cfg, model.backbone, loader, train_dataclass, val_dataclass, **trainer_kwargs)
        return (trainer, model, _), (linear_trainer, linear_model, out)
    return (trainer, model, _), _

            
@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)

    grid_search_active = omegaconf_select(cfg, "grid_search", None) and cfg.grid_search.enabled
    grid_hparams = None
    _recall_cfg = cfg.copy()
    if grid_search_active:
        grid_hparams = cfg.grid_search.hparams
        # Close the wandb and checkpointing for grid search
        cfg.wandb.enabled = False
        cfg.checkpoint.enabled = False
    else:
        grid_hparams = {"optimizer.lr": [cfg.optimizer.lr], "optimizer.weight_decay": [cfg.optimizer.weight_decay]} # placeholder, does nothing

    # Grid search by creating iterpools product of all hyperparameters
    best_hparams = {}
    best_accuracy = 0

    for values in itertools.product(*grid_hparams.values()):
        current_hparams = dict(zip(grid_hparams.keys(), values))
        # Update the configuration object with the current key-value pairs
        for key, val in current_hparams.items():
            OmegaConf.update(cfg, key, val)
        # Run the model
        result, linear_result = None, None
        if grid_search_active:
            result, linear_result = train_ssl_model(cfg, grid_search_active)
        else:
            result, _ = train_ssl_model(cfg, grid_search_active)

        if linear_result is not None:
            (_, _, out) = linear_result
            if out[0]['val_acc'] > best_accuracy:
                best_accuracy = out[0]['val_acc'] #! Adjust these for balanced acc later?
                best_hparams = current_hparams

    # Run the best model from the scratch
    print(f"Training the model wtih best hyperparameters: {best_hparams}")
    if grid_search_active:
        cfg = _recall_cfg
        cfg.wandb.enabled = _recall_cfg.wandb.enabled
        cfg.checkpoint.enabled = _recall_cfg.checkpoint.enabled

        # add the name of the model to the cfg
        for key, val in best_hparams.items():
            OmegaConf.update(cfg, key, val)

        cfg.name = f"{cfg.name}-lr-{cfg.optimizer.lr}-wd-{cfg.optimizer.weight_decay}"

        (trainer,model,_), _ = train_ssl_model(cfg, grid_search_enabled = False)
        

if __name__ == "__main__":
    main()
