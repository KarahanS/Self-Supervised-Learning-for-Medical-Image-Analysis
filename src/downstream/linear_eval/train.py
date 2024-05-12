import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch.utils.data as data

import src.utils.setup as setup
import src.utils.constants as const
from src.utils.enums import DatasetEnum
from lr import LogisticRegression


def train_logistic_regression(
    train_feats_data,
    test_feats_data,
    batch_size,
    max_epochs=100,
    **kwargs,
):
    print(kwargs)
    model = LogisticRegression(**kwargs)
    print("Logistic regression model created")

    if kwargs["log"] == "wandb":
        logger = WandbLogger(
            save_dir=const.SIMCLR_LOG_PATH,
            name=f"{kwargs['encoder']}_simclr_{kwargs['epochs']}_{kwargs['batch_size']}_pt={kwargs['pretrained']}_s={kwargs['seed']}",
            # name : display name for the run
        )  # TODO: A more sophisticated naming convention might be needed if hyperparameters are changed
        print("Logging with WandB...")
    elif kwargs["log"] == "tb":
        logger = TensorBoardLogger(save_dir=const.SIMCLR_LOG_PATH, name="tensorboard")
        print("Logging with TensorBoard...")
    else:
        print("Logging turned off.")

    # Trainer
    accelerator, num_threads = setup.get_accelerator_info()

    trainer = pl.Trainer(
        default_root_dir=const.LOGISTIC_REGRESSION_CHECKPOINT_PATH,
        accelerator=accelerator,
        devices=num_threads,
        max_epochs=max_epochs,
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
        train_loader, validation_loader, _ = loader.get_loaders()
    else:
        raise ValueError(
            "Dataset not supported yet. Please use MedMNIST."
        )  # TODO: Implement support for MIMeta

    train_loader = data.DataLoader(
        train_feats_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=const.NUM_WORKERS,
    )

    test_loader = data.DataLoader(
        test_feats_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=const.NUM_WORKERS,
    )

    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Train model
    if not use_existing_model:
        trainer.fit(model, train_loader, test_loader)

        # Load best checkpoint after training
        model = LogisticRegressionLM.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # Save model
        trainer.save_checkpoint(destination_path)

    # Test best model on train and test set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    (
        DATA_FLAG,
        MAX_EPOCHS,
        SAMPLES_PER_CLASS,
        NUM_SAMPLES,
        PRETRAINED_FILE,
        MODEL_NAME,
    ) = Arguments.parse_args_train(downstream=True)

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Load data
    downloader = Downloader()
    train_data = downloader.load(
        DATA_FLAG, SplitType.TRAIN, NUM_SAMPLES, SAMPLES_PER_CLASS
    )
    val_data = downloader.load(DATA_FLAG, SplitType.VALIDATION)
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    # Show example images
    # show_example_images(train_data, reshape=True)
    # show_example_images(val_data, reshape=True)
    # show_example_images(test_data, reshape=True)
    # sys.exit()

    model_name = MODEL_NAME or f"downstream-{DATA_FLAG}"

    # Get pretrained model
    pretrained_path = os.path.join(SIMCLR_CHECKPOINT_PATH, PRETRAINED_FILE)
    pretrained_model = get_pretrained_model(pretrained_path)

    print("Preparing data features...")

    train_feats = get_data_features_from_pretrained_model(
        pretrained_model, train_data, device
    )
    test_feats = get_data_features_from_pretrained_model(
        pretrained_model, test_data, device
    )

    print("Preparing data features: Done!")

    # Train model

    _, d = train_feats.tensors[0].shape

    model, results = train_logistic_regression(
        train_feats_data=train_feats,
        test_feats_data=test_feats,
        batch_size=min(64, len(train_data)),
        max_epochs=MAX_EPOCHS,
        feature_dim=d,
        num_classes=len(INFO[DATA_FLAG]["label"]),
        lr=1e-3,
        weight_decay=1e-3,
    )

    print(results)

    summarise()
