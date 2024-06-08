import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.enums import (
    DatasetEnum,
    SplitType,
    SSLMethod,
    DownstreamMethod,
    LoggingTools,
)
from src.ssl.simclr.train import build_simclr_model, train as train_simclr
from src.ssl.simclr.simclr import SimCLR
from src.downstream.eval.lr import build_lr
from src.downstream.eval.mlp import build_mlp
from src.utils.fileutils import create_modelname, create_ckpt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import src.utils.constants as const
import logging
from src.data.loader.medmnist_loader import MedMNISTLoader
from src.utils.config.config import Config
from src.utils.setup import get_device
from src.utils.eval import get_representations
import src.utils.setup as setup
from medmnist import INFO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# TODO : This can be done automatically! If it is a meaningful adjustment, we can create decorators for this.


MODEL_CLASS_MAP = {
    SSLMethod.SIMCLR: SimCLR,
    SSLMethod.DINO: None,
}
MODEL_BUILDER_MAP = {
    SSLMethod.SIMCLR: build_simclr_model,
    SSLMethod.DINO: None,
}

MODEL_TRAINER_MAP = {
    SSLMethod.SIMCLR: train_simclr,
    SSLMethod.DINO: None,
}

SAVE_DIR_MAP = {
    SSLMethod.SIMCLR: const.SIMCLR_LOG_PATH,
    SSLMethod.DINO: None,
}

MODEL_NAME_MAP = {
    SSLMethod.SIMCLR: "simclr",
    SSLMethod.DINO: "dino",
}

DOWNSTREAM_BUILD_MODEL_MAP = {
    DownstreamMethod.LINEAR: build_lr,
    DownstreamMethod.NONLINEAR: build_mlp,
}




class ModelWrapper:
    """
    A wrapper class for the model. This class is responsible for the following:
    - Instantiating the model
    - Configuring the optimizer and scheduler
    - Forward pass
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ssl_method = cfg.Training.Pretrain.ssl_method
        self.model = self.build_model()
        self.logger = self.build_plot_logger()
        self.train_loader, self.validation_loader = self.build_data_loaders()

    def build_model(self):
        """
        Build the model based on the configuration.
        """
        model = MODEL_BUILDER_MAP[self.ssl_method](self.cfg)
        logging.info(f"Model built successfully : {model}")
        return model

    def train_model(self):
        """
        Train the model based on the configuration.
        """
        trainer = MODEL_TRAINER_MAP[self.ssl_method]
        if trainer is None:
            logging.error(f"Training method {self.ssl_method} is not supported yet.")
            raise ValueError(f"Training method {self.ssl_method} is not supported yet.")
        logging.info(f"Training model using {self.ssl_method}...")
        return trainer(
            self.cfg, self.model, self.logger, self.train_loader, self.validation_loader
        )

    def build_plot_logger(self):

        train_params = self.cfg.Training.params
        ssl_params = self.cfg.Training.Pretrain.params
        save_dir = SAVE_DIR_MAP[self.ssl_method]

        self.modelname = create_modelname(
            ssl_params.encoder,
            train_params.max_epochs,
            train_params.batch_size,
            ssl_params.pretrained,
            self.cfg.seed,
            self.cfg.Dataset.params.image_size,
            self.cfg.Dataset.params.medmnist_flag,
            MODEL_NAME_MAP[self.ssl_method],
        )

        if self.cfg.Logging.tool == LoggingTools.WANDB:

            logger = WandbLogger(
                save_dir=save_dir,
                name=self.modelname,
                # name : display name for the run
            )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed
            logging.info("Logging with WandB...")
        elif self.cfg.Logging.tool == LoggingTools.TB:
            logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")
            logging.info("Logging with TensorBoard...")
        else:
            logger = None
            logging.info("Logging turned off.")

        return logger

    def build_data_loaders(self):
        train_params = self.cfg.Training.params
        ssl_params = self.cfg.Training.Pretrain.params

        # get train loaders
        logging.info("Preparing data loaders...")
        if self.cfg.Dataset.name == DatasetEnum.MEDMNIST:
            loader = MedMNISTLoader(
                data_flag=self.cfg.Dataset.params.medmnist_flag,
                augmentation_seq=self.cfg.Training.Pretrain.augmentations,
                download=self.cfg.Dataset.params.download,
                batch_size=train_params.batch_size,
                size=self.cfg.Dataset.params.image_size,
                num_workers=self.cfg.Device.num_workers,
            )

            train_loader = loader.get_data_and_load(
                split=SplitType.TRAIN, shuffle=True, contrastive=True
            )
            validation_loader = loader.get_data_and_load(
                split=SplitType.VALIDATION, shuffle=False, contrastive=True
            )
        else:
            raise ValueError(
                "Dataset not supported yet. Please use MedMNIST."
            )  # TODO: Implement support for MIMeta

        return train_loader, validation_loader


class DownstreamModelWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        logging.info(f"Running configuration: {cfg.convert_to_dict()}")
        self.pretrained_model = self.load_from_checkpoint()
        logging.info(f"Pretrained model loaded successfully: {self.pretrained_model}")
        logging.info(
            f"Pretrained model configuration: {self.pretrained_model.cfg_dict}"
        )
        self.pretrain_cfg = Config.convert_from_dict(self.pretrained_model.cfg_dict)
        self.logger = self.build_plot_logger()
        self.loader, self.train_dataclass, self.val_dataclass, self.test_dataclass = (
            self.build_data_loaders()
        )
        self.model, self.modelclass = self.create_downstream_model()
        logging.info(f"Downstream model created successfully: {self.model}")

    def train_model(self):
        logging.info("Preparing data features...")
        device = get_device()
        train_feats = get_representations(
            self.pretrained_model, self.train_dataclass, device
        )
        val_feats = get_representations(
            self.pretrained_model, self.val_dataclass, device
        )
        test_feats = get_representations(
            self.pretrained_model, self.test_dataclass, device
        )
        logging.info("Preparing data features: Done!")

        accelerator, num_threads = setup.get_accelerator_info()
        callback = [
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            # Auto-logs learning rate
        ]
        if self.logger is not None:
            callback.append(LearningRateMonitor("epoch"))

        trainer = pl.Trainer(
            default_root_dir=const.DOWNSTREAM_CHECKPOINT_PATH,
            accelerator=accelerator,
            devices=num_threads,
            max_epochs=self.cfg.Training.params.max_epochs,
            logger=self.logger,
            callbacks=(callback),
        )

        # Do not require optional logging
        if self.logger is not None:
            trainer.logger._default_hp_metric = None

        train_loader = self.loader.load(train_feats, shuffle=True)
        validation_loader = self.loader.load(val_feats, shuffle=False)
        test_loader = self.loader.load(test_feats, shuffle=False)

        trainer.fit(self.model, train_loader, validation_loader)
        # Load best checkpoint after training
        model = self.modelclass.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # Save model
        # TODO: save_steps is given in config but not used
        ckpt = create_ckpt(const.DOWNSTREAM_CHECKPOINT_PATH, self.modelname)
        trainer.save_checkpoint(ckpt)

        # Test model
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        test_acc = test_result[0]["test_acc"]

        logging.info(test_acc)
        return model, test_acc

    def load_from_checkpoint(self):
        eval_params = self.cfg.Training.Downstream.params
        LightningModel = MODEL_CLASS_MAP[self.cfg.Training.Downstream.ssl_method]
        logging.info(f"Loading model from checkpoint: {eval_params.pretrained_path}")
        pretrained_model = LightningModel.load_from_checkpoint(
            eval_params.pretrained_path, strict=False
        )
        return pretrained_model

    def create_downstream_model(self):
        model = DOWNSTREAM_BUILD_MODEL_MAP[self.cfg.Training.Downstream.eval_method](
            self.cfg, self.pretrain_cfg, self.loader.get_num_classes()
        )
        modelclass = model.__class__
        return model, modelclass

    def build_plot_logger(self):
        train_params = self.cfg.Training.params
        eval_params = self.cfg.Training.Downstream.params

        self.modelname = create_modelname(
            eval_params.encoder,
            train_params.max_epochs,
            train_params.batch_size,
            eval_params.pretrained,
            self.cfg.seed,
            self.cfg.Dataset.params.image_size,
            self.cfg.Dataset.params.medmnist_flag,
            self.cfg.Training.Downstream.ssl_method,
            self.cfg.Training.Downstream.eval_method,
        )

        if self.cfg.Logging.tool == LoggingTools.WANDB:
            logger = WandbLogger(
                save_dir=const.DOWNSTREAM_LOG_PATH,
                name=self.modelname,
                # name: display name for the run
            )
            logging.info("Logging with WandB...")
        elif self.cfg.Logging.tool == LoggingTools.TB:
            logger = TensorBoardLogger(
                save_dir=const.DOWNSTREAM_LOG_PATH, name="tensorboard"
            )
            logging.info("Logging with TensorBoard...")
        else:
            logging.info("Logging turned off.")
            logger = None

        return logger

    def build_data_loaders(self):
        train_params = self.cfg.Training.params

        # get train loaders
        logging.info("Preparing data loaders...")

        if self.cfg.Dataset.name == DatasetEnum.MEDMNIST:
            loader = MedMNISTLoader(
                data_flag=self.cfg.Dataset.params.medmnist_flag,
                augmentation_seq=self.cfg.Training.Downstream.augmentations,
                download=self.cfg.Dataset.params.download,
                batch_size=train_params.batch_size,
                size=self.cfg.Dataset.params.image_size,
                num_workers=self.cfg.Device.num_workers,
            )

            train_dataclass = loader.get_data(SplitType.TRAIN)
            val_dataclass = loader.get_data(SplitType.VALIDATION)
            test_dataclass = loader.get_data(
                SplitType.TEST
            )  # to be used afterwards for testing
        else:
            logging.error("Dataset not supported yet. Please use MedMNIST.")
            raise ValueError(
                "Dataset not supported yet. Please use MedMNIST."
            )  # TODO: Implement support for MIMeta
        return loader, train_dataclass, val_dataclass, test_dataclass


# TODO : Add support for DINO
# TODO : Wrapper for Downstream tasks
