import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchmetrics import AUROC


# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        """
        Single linear layer PyTorch Lightning module

        Args:
            feature_dim (int): The dimensionality of the input features.
            num_classes (int): The number of classes in the classification task.
            lr (float): The learning rate.
            weight_decay (float): The weight decay for AdamW optimiser.
            max_epochs (int, optional): The maximum number of epochs to train.
                Defaults to 100.

        """
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)
        self.auroc = AUROC(task="multiclass", num_classes=self.hparams.num_classes)

    def configure_optimizers(self):
        """
        Lightning Module utility method. Using AdamW optimiser with MultiStepLR
        scheduler. Do not call this method.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.hparams.max_epochs * 0.6),
                int(self.hparams.max_epochs * 0.8),
            ],
            gamma=0.1,
        )

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        """
        Performs forward pass on the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data.
        """
        return self.model(x)

    def loss(self, y, y_pred):
        """
        Computes the cross-entropy loss.

        Args:
            y (torch.Tensor): The target labels.
            y_pred (torch.Tensor): The predicted labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return F.cross_entropy(y_pred, y.long())

    def step(self, batch, mode="train"):
        """
        Performs a forward pass for a given batch. This method should not be
        called. Use fit() instead.
        """
        x, y = batch

        y_pred = self.forward(x)

        loss = self.loss(y, y_pred)
        acc = (y_pred.argmax(dim=-1) == y).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)

        return loss

    def training_step(self, batch, batch_index):
        """
        Performs a forward training pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_index):
        """
        Performs a forward validation pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        self.step(batch, mode="val")

    def test_step(self, batch, batch_index):
        """
        Performs a forward test pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        self.step(batch, mode="test")


def build_lr(cfg, pretrain_cfg, num_classes):
    """
    Builds a logistic regression model.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        LogisticRegression: The logistic regression model.
    """
    model = LogisticRegression(
        feature_dim=pretrain_cfg.Training.Pretrain.params.output_dim,
        num_classes=num_classes,
        lr=cfg.Training.params.learning_rate,
        weight_decay=cfg.Training.params.weight_decay,
        max_epochs=cfg.Training.params.max_epochs,
    )

    return model
