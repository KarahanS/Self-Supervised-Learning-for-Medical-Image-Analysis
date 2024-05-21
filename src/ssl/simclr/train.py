from pytorch_lightning import Trainer
from src.ssl.simclr.simclr import SimCLR
from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
import src.utils.constants as const
from src.utils.enums import DatasetEnum, SplitType, LoggingTools

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
import torch
import torchvision.models as models


def train(cfg):
    if cfg.Logging.tool == LoggingTools.WANDB:
        train_params = cfg.Training.params
        ssl_params = cfg.Training.Pretrain.params

        logger = WandbLogger(
            save_dir=const.SIMCLR_LOG_PATH,
            name=f"{ssl_params.encoder}_simclr_{train_params.epochs}_{train_params.batch_size}_pt={ssl_params.pretrained}" \
                f"_s={cfg.seed}_img={cfg.Dataset.params.image_size}",
            # name : display name for the run
        )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed
        print("Logging with WandB...")
    elif cfg.Logging.tool == LoggingTools.TB:
        logger = TensorBoardLogger(save_dir=const.SIMCLR_LOG_PATH, name="tensorboard")
        print("Logging with TensorBoard...")
    else:
        logger = None
        print("Logging turned off.")

    # Define the encoder
    if ssl_params.encoder not in models.list_models():
        raise ValueError(
            "Encoder not found among the available torchvision models. Please make sure that you have entered the correct model name."
        )
        ## TODO: Add support for custom models
    if ssl_params.pretrained:  # TODO: Implement support for pretrained models - weights can be stored as enum
        encoder = models.get_model(ssl_params.encoder, weights="IMAGENET1K_V2")
    else:
        encoder = models.get_model(ssl_params.encoder, weights=None)

    feature_size = encoder.fc.in_features
    encoder.fc = (
        torch.nn.Identity()
    )  # Replace the fully connected layer with identity function

    # Define the model
    model = SimCLR(
        encoder=encoder,
        n_views=ssl_params.n_views,
        feature_size=feature_size,
        hidden_dim=ssl_params.hidden_dim,
        output_dim=ssl_params.output_dim,
        weight_decay=train_params.weight_decay,
        lr=train_params.lr,
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
        max_epochs=train_params.epochs,
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

    ckpt = (
        const.SIMCLR_CHECKPOINT_PATH
        + f"{ssl_params.encoder}_simclr_{train_params.epochs}_{train_params.batch_size}_pt={ssl_params.pretrained}" \
            f"_s={cfg.seed}_img={cfg.Dataset.params.image_size}.ckpt"
    )
    # Save pretrained model
    trainer.save_checkpoint(ckpt)
    timer.time_elapsed("train")
    timer.start_time("validate")
    timer.end_time("test")

    return model
