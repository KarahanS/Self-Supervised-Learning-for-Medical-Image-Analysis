from pytorch_lightning import Trainer
from src.ssl.simclr.simclr import SimCLR
from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
import src.utils.constants as const
from src.utils.enums import DatasetEnum, SplitType, LoggingTools
from src.utils.fileutils import create_modelname, create_ckpt

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
import logging


def build_simclr_model(cfg):
    # Define the model
    train_params = cfg.Training.params
    ssl_params = cfg.Training.Pretrain.params

    model = SimCLR(
        encoder=ssl_params.encoder,
        n_views=ssl_params.n_views,
        pretrained=ssl_params.pretrained,
        hidden_dim=ssl_params.hidden_dim,
        output_dim=ssl_params.output_dim,
        weight_decay=train_params.weight_decay,
        lr=train_params.learning_rate,
        temperature=ssl_params.temperature,
        max_epochs=train_params.max_epochs,
        cfg_dict=cfg.convert_to_dict(),  # So that we can save the cfg
    )

    return model


def train(cfg, model, logger, train_loader, validation_loader):
    train_params = cfg.Training.params
    ssl_params = cfg.Training.Pretrain.params

    modelname = create_modelname(
        ssl_params.encoder,
        train_params.max_epochs,
        train_params.batch_size,
        ssl_params.pretrained,
        cfg.seed,
        cfg.Dataset.params.image_size,
        cfg.Dataset.params.medmnist_flag,
        "simclr",
    )

    accelerator, num_threads = setup.get_accelerator_info()

    # timer
    timer = Timer(duration="00:72:00:00")

    callback = [
        # Save model as checkpoint periodically under checkpoints folder
        ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc_top5"),
        # Auto-logs learning rate
        timer,
    ]

    if logger is not None:
        callback.append(LearningRateMonitor("epoch"))

    trainer = Trainer(
        default_root_dir=const.SIMCLR_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=train_params.max_epochs,
        logger=logger,
        callbacks=(callback),
    )

    # Train the model
    trainer.fit(model, train_loader, validation_loader)

    # Load best checkpoint after training
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # TODO: save_steps is given in config but not used
    ckpt = create_ckpt(const.SIMCLR_CHECKPOINT_PATH, modelname)
    # Save pretrained model
    trainer.save_checkpoint(ckpt)
    timer.time_elapsed("train")
    timer.start_time("validate")
    timer.end_time("test")

    return model
