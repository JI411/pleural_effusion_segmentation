"""
Base class for all models.
"""

import segmentation_models_pytorch as smp
import torch


class BaseModel(torch.nn.Module):
    """Base model class used in typing and hinting."""

    def __init__(self, in_channels: int = 1, **kwargs):
        """Create model."""
        super().__init__()
        self.in_channels = in_channels
        self.kwargs = kwargs

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run model on image/batch."""
        return self.model(image)


class BaseUnetSMPModel(BaseModel):
    """Wrapper for smp.Unet model."""

    in_channels: int = 1

    def __init__(self, encoder_name: str = 'resnet34') -> None:
        """Create segmentation_models_pytorch.Unet model."""
        super().__init__(in_channels=self.in_channels)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            decoder_channels=[256, 128, 64, 32, 64],
            in_channels=self.in_channels,
            classes=self.in_channels,
            aux_params=None,
            activation=None
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run model on batch."""
        raise NotImplementedError
