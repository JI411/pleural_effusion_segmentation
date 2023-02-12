"""
Main training script
"""
# pylint: disable=unused-argument, arguments-differ
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.dataset import Batch
from src.data.batching import get_standard_dataloaders
from src.dice import BinaryDiceLoss
from src.model.wrappers import BaseModel


class PleuralSegmentationModule(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """ Lightning wrapper for models, connect loss, dataloader and model """

    def __init__(self, model: BaseModel, batch_size: int) -> None:
        """ Create model for training """
        super().__init__()

        self.model = model
        self.loss = BinaryDiceLoss()
        self.batch_size = batch_size
        self.loaders = get_standard_dataloaders(
            batch_size=self.batch_size, train_transforms=model.train_transforms, valid_transforms=model.valid_transforms
        )

    def training_step(self, batch: Batch, batch_idx: int) -> float:
        """ Train model on batch """
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask'])
        self.log("train_loss", score)
        return score

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """ Validate model on batch """
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask']).item()
        self.log("test_loss", score)

    def train_dataloader(self) -> DataLoader:
        """ Get train dataloader """
        return self.loaders.train

    def val_dataloader(self) -> DataLoader:
        """ Get validation dataloader """
        return self.loaders.valid

    def configure_optimizers(self):
        """ Configure optimizer """
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
