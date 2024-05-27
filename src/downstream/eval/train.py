from medmnist import INFO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
from src.utils.fileutils import create_modelname, create_ckpt
from src.utils.setup import get_device
import src.utils.constants as const
from src.utils.enums import (
    DatasetEnum,
    SplitType,
    SSLMethod,
    DownstreamMethod,
    LoggingTools,
)
from src.downstream.eval.lr import LogisticRegression
from src.downstream.eval.mlp import MultiLayerPerceptron

from src.ssl.simclr.simclr import SimCLR

from src.utils.eval import get_auroc_metric, get_representations

import os
import datetime
import logging

def train(cfg):
    train_params = cfg.Training.params
    eval_params = cfg.Training.Downstream.params
    modelname = create_modelname(
        eval_params.encoder,
        train_params.max_epochs,
        train_params.batch_size,
        eval_params.pretrained,
        cfg.seed,
        cfg.Dataset.params.image_size,
        cfg.Dataset.params.medmnist_flag,
        cfg.Training.Downstream.ssl_method,
        cfg.Training.Downstream.eval_method,
    )

    # get train loaders
    if cfg.Dataset.name == DatasetEnum.MEDMNIST:
        loader = MedMNISTLoader(
            data_flag=cfg.Dataset.params.medmnist_flag,
            augmentation_seq=cfg.Training.Downstream.augmentations,
            download=cfg.Dataset.params.download,
            batch_size=train_params.batch_size,
            size=cfg.Dataset.params.image_size,
            num_workers=cfg.Device.num_workers,
        )

        train_dataclass = loader.get_data(SplitType.TRAIN)
        val_dataclass = loader.get_data(SplitType.VALIDATION)
        test_dataclass = loader.get_data(
            SplitType.TEST
        )  # to be used afterwards for testing

    else:
        raise ValueError(
            "Dataset not supported yet. Please use MedMNIST."
        )  # TODO: Implement support for MIMeta

        model_name = ...

    if (
        cfg.Training.Downstream.ssl_method == SSLMethod.SIMCLR
    ):  # TODO: Take the model from a dictionary rather than if-else
        LightningModel = SimCLR
    else:
        raise ValueError("Other SSL methods are not supported yet.")
    pretrained_model = LightningModel.load_from_checkpoint(
        eval_params.pretrained_path, strict=False
    )

    logging.info("Preparing data features...")
    device = get_device()
    train_feats = get_representations(pretrained_model, train_dataclass, device)
    val_feats = get_representations(pretrained_model, val_dataclass, device)
    test_feats = get_representations(pretrained_model, test_dataclass, device)

    logging.info("Preparing data features: Done!")

    # Train model

    _, d = train_feats.tensors[0].shape

    if cfg.Training.Downstream.eval_method == DownstreamMethod.LINEAR:
        model = LogisticRegression(
            feature_dim=d,
            num_classes=loader.get_num_classes(),
            lr=train_params.learning_rate,
            weight_decay=train_params.weight_decay,
            max_epochs=train_params.max_epochs,
        )
        modelclass = LogisticRegression
    elif cfg.Training.Downstream.eval_method == DownstreamMethod.NONLINEAR:
        model = MultiLayerPerceptron(
            feature_dim=d,
            hidden_dim=eval_params.hidden_dim,
            num_classes=loader.get_num_classes(),
            lr=train_loader.lr,
            weight_decay=train_params.weight_decay,
            max_epochs=train_params.max_epochs,
        )
        modelclass = MultiLayerPerceptron

    # TODO: Log every n steps is given in config but not used
    if cfg.Logging.tool == LoggingTools.WANDB:
        logger = WandbLogger(
            save_dir=const.DOWNSTREAM_LOG_PATH,
            name=modelname,
            # name: display name for the run
        )
        logging.info("Logging with WandB...")
    elif cfg.Logging.tool == LoggingTools.TB:
        logger = TensorBoardLogger(
            save_dir=const.DOWNSTREAM_LOG_PATH, name="tensorboard"
        )
        logging.info("Logging with TensorBoard...")
    else:
        logging.info("Logging turned off.")
        logger = None

    # Trainer
    accelerator, num_threads = setup.get_accelerator_info()
    callback = [
        # Save model as checkpoint periodically under checkpoints folder
        ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
        # Auto-logs learning rate
    ]
    if logger is not None:
        callback.append(LearningRateMonitor("epoch"))

    trainer = pl.Trainer(
        default_root_dir=const.DOWNSTREAM_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=train_params.max_epochs,
        logger=logger,
        callbacks=(callback),
    )

    # Do not require optional logging
    if logger is not None:
        trainer.logger._default_hp_metric = None

    train_loader = loader.load(train_feats, shuffle=True)
    validation_loader = loader.load(val_feats, shuffle=False)
    test_loader = loader.load(test_feats, shuffle=False)

    trainer.fit(model, train_loader, validation_loader)
    # Load best checkpoint after training
    model = modelclass.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Save model
    # TODO: save_steps is given in config but not used
    ckpt = create_ckpt(const.DOWNSTREAM_CHECKPOINT_PATH, modelname)
    trainer.save_checkpoint(ckpt)

    # Test model
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    if cfg.Dataset.name == DatasetEnum.MEDMNIST:
        data_flag = cfg.Dataset.params.medmnist_flag.value

        result = {
            "test accuracy": test_result[0]["test_acc"],
            "auroc": get_auroc_metric(
                model, test_loader, num_classes=len(INFO[data_flag]["label"])
            ),
        }
    else:
        RuntimeError("Dataset not supported yet. Please use MedMNIST.")

    logging.info(result)
    return model, result
