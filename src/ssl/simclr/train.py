from pytorch_lightning import Trainer
from src.ssl.simclr.simclr import SimCLR
from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
import src.utils.constants as const
from src.utils.enums import DatasetEnum
from src.utils.enums import SplitType

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
import torch
import torchvision.models as models


# args will be a tuple, and kwargs will be a dict.
def train(*args, **kwargs):
    print(kwargs)

    if kwargs["log"] == "wandb":
        logger = WandbLogger(
            save_dir=const.SIMCLR_LOG_PATH,
            name=f"{kwargs['encoder']}_simclr_{kwargs['epochs']}_{kwargs['batch_size']}_pt={kwargs['pretrained']}_s={kwargs['seed']}_img={kwargs['size']}",
            # name : display name for the run
        )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed
        print("Logging with WandB...")
    elif kwargs["log"] == "tb":
        logger = TensorBoardLogger(save_dir=const.SIMCLR_LOG_PATH, name="tensorboard")
        print("Logging with TensorBoard...")
    else:
        print("Logging turned off.")

    # Define the encoder
    if kwargs["encoder"] not in models.list_models():
        raise ValueError(
            "Encoder not found among the available torchvision models. Please make sure that you have entered the correct model name."
        )
        ## TODO: Add support for custom models
    if kwargs[
        "pretrained"
    ]:  # TODO: Implement support for pretrained models - weights can be stored as enum
        encoder = models.get_model(kwargs["encoder"], weights="IMAGENET1K_V2")
    else:
        encoder = models.get_model(kwargs["encoder"], weights=None)

    feature_size = encoder.fc.in_features
    encoder.fc = (
        torch.nn.Identity()
    )  # Replace the fully connected layer with identity function

    # Define the model
    model = SimCLR(
        encoder=encoder,
        feature_size=feature_size,
        hidden_dim=kwargs["hidden_dim"],
        output_dim=kwargs["output_dim"],
        weight_decay=kwargs["weight_decay"],
        lr=kwargs["lr"],
    )

    accelerator, num_threads = setup.get_accelerator_info()

    # timer
    timer = Timer(duration="00:72:00:00")

    trainer = Trainer(
        default_root_dir=const.SIMCLR_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=kwargs["epochs"],
        logger=logger,
        callbacks=[
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(
                save_weights_only=False, mode="max", monitor="val_acc_top5"
            ),
            # Auto-logs learning rate
            LearningRateMonitor("epoch"),
            timer,
        ],
    )

    # get train loaders
    if kwargs["dataset_name"] == DatasetEnum.MEDMNIST:
        loader = MedMNISTLoader(
            data_flag=kwargs["data_flag"],
            augmentation_seq=kwargs["augmentation"],
            download=True,
            batch_size=kwargs["batch_size"],
            size=kwargs["size"],
            num_workers=kwargs["num_workers"],
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

    # Save pretrained model
    trainer.save_checkpoint(const.SIMCLR_CHECKPOINT_PATH + f"{kwargs['encoder']}.ckpt")
    timer.time_elapsed("train")
    timer.start_time("validate")
    timer.end_time("test")

    return model
