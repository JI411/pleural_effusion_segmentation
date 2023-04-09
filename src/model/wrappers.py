"""
Models for segmentation.
"""
from enum import Enum, unique
from typing import Type

from src.model.base import BaseModelWrapper
from src.model.model_zoo.unet_3d import UNet3D


class UNet3DWrapper(BaseModelWrapper):
    """Wrapper for 3D UNet model."""

    def __init__(self, base_n_filter: int = 8):
        """Create 3D UNet model."""
        super().__init__(in_channels=1, n_classes=1)
        self.base_n_filter = base_n_filter
        self.model = UNet3D(in_channels=self.in_channels, n_classes=self.n_classes, base_n_filter=self.base_n_filter)

# pylint: disable=invalid-name
@unique
class ModelsZoo(Enum):
    """Models available in the model zoo."""
    unet3d: Type[BaseModelWrapper] = UNet3DWrapper
