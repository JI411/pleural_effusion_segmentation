"""
Main training script
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.dataset import Batch
from src.data.batching import get_standard_dataloaders
from src.dice import BinaryDiceLoss
from src.model.wrappers import BaseModel


class PleuralSegmentationModule(pl.LightningModule):
    """ Lightning wrapper for models, connect loss, dataloader and model """

    def __init__(self, model: BaseModel, batch_size) -> None:
        """ Create model from segmentation_models_pytorch """
        super().__init__()

        self.model = model
        self.loss = BinaryDiceLoss()
        self.batch_size = batch_size
        self.loaders = get_standard_dataloaders(batch_size=self.batch_size)

    def training_step(self, batch: Batch, batch_idx: int) -> float:
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask'])
        self.log("train_loss", score)
        return score

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask'])
        self.log("test_loss", score)

    def train_dataloader(self) -> DataLoader:
        return self.loaders.train

    def val_dataloader(self) -> DataLoader:
        return self.loaders.valid

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

