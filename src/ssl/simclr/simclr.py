import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

from src.ssl.losses import nt_xent


# https://pytorch-lightning.readthedocs.io/en/0.7.6/lightning-module.html
class SimCLR(pl.LightningModule):
    def __init__(
        self,
        encoder,
        n_views,
        hidden_dim,
        feature_size,
        output_dim,
        weight_decay,
        lr,
        max_epochs=200,
        temperature=0.05,
    ):
        """
        SimCLR model

        Args:
            encoder: Encoder network
            hidden_dim: Hidden dimension of the projection head
            output_dim: Output dimension of the projection head
            weight_decay: Weight decay for the AdamW optimizer
            similarity: Similarity metric to use
            temperature: Temperature parameter for the softmax in NT-Xent loss
        """
        super(SimCLR, self).__init__()
        # save constructor parameters to self.hparams
        self.save_hyperparameters()

        self.encoder = encoder  # base encoder without projection head
        # Define your projection head
        self.projector = Projector(
            input_dim=self.hparams.feature_size,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.output_dim,
        )

    def configure_optimizers(self):
        """
        Lightning Module utility method. Using AdamW optimiser with
        CosineAnnealingLR scheduler.
        """
        # AdamW decouples weight decay from gradient updates
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # TODO: LARS implementation:  (better for large batch sizes)
        # base_optimizer = optim.AdamW(model.parameters(), lr=0.1)
        # optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

        # Set learning rate using a cosine annealing schedule
        # See https://pytorch.org/docs/stable/optim.html
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50,
        )

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        return self.projector(self.encoder(x))

    def step(self, batch, mode="train"):
        """
        Performs a forward pass for a given batch. This method should not be
        called. Use fit() instead.

        NT-Xent is used.
        """
        # len(batch) = 2
        x, _ = batch
        x = torch.cat(x, dim=0)  # x[0].shape = (3, 28, 28)   (list of images)

        # Apply base encoder and projection head to images to get embedded
        # encoders
        z = self.forward(x)
        loss, sim_argsort = nt_xent(z, temperature=self.hparams.temperature)

        # Logging loss
        self.log(mode + "_loss", loss)

        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return loss

    def training_step(self, batch, batch_index):
        """
        Performs a forward training pass for a given batch. Lightning Module
        utility method.
        """
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_index):
        """
        Performs a forward validation pass for a given batch. Lightning Module
        utility method.
        """
        self.step(batch, mode="val")


# projection head (takes the output of the encoder and projects it to a lower-dimensional space)
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)
