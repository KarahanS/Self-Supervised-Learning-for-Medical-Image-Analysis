import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F


# https://pytorch-lightning.readthedocs.io/en/0.7.6/lightning-module.html
class SimCLR(pl.LightningModule):
    def __init__(
        self,
        encoder,
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
        self.save_hyperparameters(ignore=["encoder"])

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

        InfoNCE Loss is implemented.
        """
        # len(batch) = 2
        x, _ = batch
        x = torch.cat(x, dim=0)

        # Apply base encoder and projection head to images to get embedded
        # encoders
        z = self.forward(x)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        cos_sim /= self.hparams.temperature

        # InfoNCE loss
        loss = (-cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)).mean()

        # Logging loss
        self.log(mode + "_loss", loss)

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            # First position positive example
            dim=-1,
        )

        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

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
