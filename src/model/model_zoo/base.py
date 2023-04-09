"""
Base model class from https://github.com/black0017/MedicalZooPytorch/tree/master/lib/medzoo (MIT License).
Original repository have many dependencies, that are not needed for this project.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """ BaseModel for classes from MedicalZooPytorch."""

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    def count_params(self):
        """ Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        """Run model on image/batch."""
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()
