"""
Base class for all models.
"""

import torch


class BaseModelWrapper(torch.nn.Module):
    """Base model class used in typing and hinting."""

    def __init__(self, in_channels: int = 1, n_classes: int = 1, **kwargs):
        """Create model."""
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.kwargs = kwargs

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run model on image/batch."""
        return self.model(image)
