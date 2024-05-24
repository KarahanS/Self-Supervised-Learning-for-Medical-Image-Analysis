from pytorch_lightning import Trainer
from src.ssl.simclr.simclr import SimCLR
from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
import src.utils.constants as const
from src.utils.enums import DatasetEnum, SplitType, LoggingTools
from src.utils.fileutils import create_modelname, create_ckpt

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer


def train(cfg):
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
    # TODO: Log every n steps is given in config but not used
    if cfg.Logging.tool == LoggingTools.WANDB:

        logger = WandbLogger(
            save_dir=const.SIMCLR_LOG_PATH,
            name=modelname,
            # name : display name for the run
        )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed
        print("Logging with WandB...")
    elif cfg.Logging.tool == LoggingTools.TB:
        logger = TensorBoardLogger(save_dir=const.SIMCLR_LOG_PATH, name="tensorboard")
        print("Logging with TensorBoard...")
    else:
        logger = None
        print("Logging turned off.")

    # Define the model
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

    # get train loaders
    if cfg.Dataset.name == DatasetEnum.MEDMNIST:
        loader = MedMNISTLoader(
            data_flag=cfg.Dataset.params.medmnist_flag,
            augmentation_seq=cfg.Training.Pretrain.augmentations,
            download=cfg.Dataset.params.download,
            batch_size=train_params.batch_size,
            size=cfg.Dataset.params.image_size,
            num_workers=cfg.Device.num_workers,
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
