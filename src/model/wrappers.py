"""
Models for segmentation
"""

import torch
from pytorch3dunet.unet3d.model import UNet2D, UNet3D, ResidualUNet3D

from src.model.base import BaseModel, BaseUnetSMPModel


class Unet2DWrapper(BaseModel):
    """ Wrapper for 2D UNet model """

    def __init__(self, in_channels: int = 1):
        """ Create 2D UNet model """
        super().__init__(in_channels=in_channels)
        self.model = UNet2D(in_channels=in_channels, out_channels=in_channels, final_sigmoid=False)


class Unet3DWrapper(BaseModel):
    """ Wrapper for 3D UNet model """

    def __init__(self):
        """ Create 3D UNet model """
        super().__init__(in_channels=1)
        self.model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=False)


class ResidualUNet3DWrapper(BaseModel):
    """ Wrapper for 3D Residual UNet model """

    def __init__(self):
        """ Create 3D Residual UNet model """
        super().__init__(in_channels=1)
        self.model = ResidualUNet3D(in_channels=1, out_channels=1, final_sigmoid=False)


class UnetSMP2DWrapper(BaseUnetSMPModel):
    """ Wrapper for smp.Unet model """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """ Run model on tensor with shape (batch, 1, height, width) """
        return self.model(image)

class UnetSMP3DWrapper(BaseUnetSMPModel):
    """ Wrapper for smp.Unet model """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run model on tensor with shape (batch, 1, channels, height, width).

        Run on every channel separately and accumulate results.
        """
        result = []
        for tensor_slice in torch.moveaxis(image, 2, 0):
            pred = self.model(tensor_slice)
            result.append(pred)

        result = torch.cat(result, dim=1)
        return result.unsqueeze(1)
