from medmnist import INFO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.loader.medmnist_loader import MedMNISTLoader
import src.utils.setup as setup
from src.utils.setup import get_device
import src.utils.constants as const
from src.utils.enums import DatasetEnum
from src.utils.enums import SplitType
from src.downstream.eval.lr import LogisticRegression
from src.downstream.eval.mlp import MultiLayerPerceptron

from src.ssl.simclr.simclr import SimCLR
import pytorch_lightning as pl
from src.utils.eval import get_auroc_metric, get_representations

import os


# helper to save model without overwriting previous runs
def save_without_overwrite(path):
    inc = 0
    while os.path.exists(path):
        # update ckpt name
        inc += 1
        path = path + f"_{inc}"
    return path


def train(*args, **kwargs):
    print(kwargs)

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

        train_dataclass = loader.get_data(SplitType.TRAIN)
        val_dataclass = loader.get_data(SplitType.VALIDATION)
        test_dataclass = loader.get_data(
            SplitType.TEST
        )  # to be used afterwards for testing
    else:
        raise ValueError(
            "Dataset not supported yet. Please use MedMNIST."
        )  # TODO: Implement support for MIMeta

    model_name = f"{kwargs['eval_method']}_{kwargs['encoder']}_{kwargs['data_flag']}"

    if (
        kwargs["ssl_method"] == "simclr"
    ):  # TODO: Take the model from a dictionary rather than if-else
        LightningModel = SimCLR
    else:
        raise ValueError("Other SSL methods are not supported yet.")
    pretrained_model = LightningModel.load_from_checkpoint(
        kwargs["pretrained_path"], strict=False
    )

    print("Preparing data features...")
    device = get_device()
    train_feats = get_representations(pretrained_model, train_dataclass, device)
    val_feats = get_representations(pretrained_model, val_dataclass, device)
    test_feats = get_representations(pretrained_model, test_dataclass, device)

    print("Preparing data features: Done!")

    # Train model

    _, d = train_feats.tensors[0].shape

    if kwargs["eval_method"] == "linear":
        model = LogisticRegression(
            feature_dim=d,
            num_classes=loader.get_num_classes(),
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
            max_epochs=kwargs["epochs"],
        )
        modelclass = LogisticRegression
    elif kwargs["eval_method"] == "nonlinear":
        model = MultiLayerPerceptron(
            feature_dim=d,
            hidden_dim=kwargs["hidden_dim"],
            num_classes=loader.get_num_classes(),
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
            max_epochs=kwargs["epochs"],
        )
        modelclass = MultiLayerPerceptron

    print("Logistic regression model created")

    if kwargs["log"] == "wandb":
        logger = WandbLogger(
            save_dir=const.DOWNSTREAM_LOG_PATH,
            name=f"{model_name}",
            # name: display name for the run
        )
        print("Logging with WandB...")
    elif kwargs["log"] == "tb":
        logger = TensorBoardLogger(
            save_dir=const.DOWNSTREAM_LOG_PATH, name="tensorboard"
        )
        print("Logging with TensorBoard...")
    else:
        print("Logging turned off.")

    # Trainer
    accelerator, num_threads = setup.get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=const.DOWNSTREAM_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=kwargs["epochs"],
        logger=logger,
        callbacks=[
            # Save model as checkpoint periodically under checkpoints folder
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc"),
            # Auto-logs learning rate
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=10,
    )

    # Do not require optional logging
    trainer.logger._default_hp_metric = None

    train_loader = loader.load(train_feats, shuffle=True)
    validation_loader = loader.load(val_feats, shuffle=False)
    test_loader = loader.load(test_feats, shuffle=False)

    trainer.fit(model, train_loader, validation_loader)

    # Load best checkpoint after training
    model = modelclass.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Save model
    ckpt = save_without_overwrite(
        const.DOWNSTREAM_CHECKPOINT_PATH + f"{model_name}.ckpt"
    )
    trainer.save_checkpoint(ckpt)

    # Test model
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    data_flag = kwargs["data_flag"].value

    result = {
        "top-1 acc": test_result[0]["test_acc"],
        "auroc": get_auroc_metric(
            model, test_loader, num_classes=len(INFO[data_flag]["label"])
        ),
    }

    print(result)
    return model, result
