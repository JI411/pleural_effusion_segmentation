"""
Main training script.
"""
# pylint: disable=unused-argument, arguments-differ
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import const
from src.data import preprocessing
from src.data.base import Sample
from src.data.dataset import PleuralEffusionDataset3D
from src.loss.dice import BinaryDiceLoss
from src.model.wrappers import BaseModelWrapper


class PleuralSegmentationModule(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """Lightning wrapper for models, connect loss, dataloader and model."""

    def __init__(self, model: BaseModelWrapper, batch_size: int, ) -> None:
        """Create model for training."""
        super().__init__()

        self.model = model
        self.loss = BinaryDiceLoss()
        self.batch_size = batch_size
        self.num_workers = const.DEFAULT_NUM_WORKERS

    def training_step(self, batch: Sample, batch_idx: int) -> float:
        """Train model on batch."""
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask'])
        self.log("train_loss", score)
        return score

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        """Validate model on batch."""
        predict = self.model(batch['image'])
        score = self.loss.forward(raw_logits=predict, mask=batch['mask']).item()
        self.log("test_loss", score)

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        dataset = PleuralEffusionDataset3D(
            data_dir=const.DatasetPathConfig.train_dir,
            augmentation=preprocessing.train_augmentation(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        dataset = PleuralEffusionDataset3D(
            data_dir=const.DatasetPathConfig.valid_dir,
            augmentation=preprocessing.valid_augmentation(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
