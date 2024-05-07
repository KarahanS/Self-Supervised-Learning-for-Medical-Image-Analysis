import torch
from torchvision.models import resnet18
from pytorch_lightning import Trainer
from ssl.simclr.simclr import SimCLR
from loader.medmnist_loader import MedMNISTLoader
import torchvision.models as models

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import utils.setup as setup
import utils.constants as const
from utils.enums import DatasetEnum


# args will be a tuple, and kwargs will be a dict.
def train(*args, **kwargs):
    print(kwargs)

    # Define the encoder
    if kwargs["encoder"] not in models.list_models():
        raise ValueError(
            "Encoder not found among the available torchvision models. Please make sure that you have entered the correct model name."
        )
        ## TODO: Add support for custom models
    encoder = models.get_model(kwargs["encoder"], pretrained=False)
    feature_size = encoder.fc.in_features
    encoder.fc = (
        torch.nn.Identity()
    )  # Remove the final fully connected layer and replace it with identity function

    logger = WandbLogger(
        save_dir=const.SIMCLR_TB_PATH, name=f"{kwargs['encoder']}_simclr"
    )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed

    # Define the model
    model = SimCLR(
        encoder=kwargs["encoder"],
        hidden_dim=kwargs["hidden_dim"],
        feature_size=feature_size,
        output_dim=kwargs["output_dim"],
        weight_decay=kwargs["weight_decay"],
        lr=kwargs["lr"],
    )

    accelerator, num_threads = setup.get_accelerator_info()

    trainer = Trainer(
        default_root_dir=const.SIMCLR_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=kwargs["max_epochs"],
        logger=logger,
        callbacks=[
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(
                save_weights_only=False, mode="max", monitor="val_acc_top5"
            ),
            # Auto-logs learning rate
            LearningRateMonitor("epoch"),
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
        )
        train_loader, validation_loader, _ = loader.get_loaders()
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

    return model
