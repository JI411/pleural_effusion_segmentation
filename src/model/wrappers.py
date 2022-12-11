"""
Models for segmentation
"""

import segmentation_models_pytorch as smp
from pytorch3dunet.unet3d.model import UNet2D, UNet3D, ResidualUNet3D

from src.model.base import BaseModel


class Unet2DWrapper(BaseModel):
    """ Wrapper for 2D UNet model """

    def __init__(self, in_channels: int):
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


class UnetSMPWrapper(BaseModel):
    """ Wrapper for smp.Unet model """
    def __init__(self, in_channels: int, encoder_name: str = 'resnet34') -> None:
        """ Create segmentation_models_pytorch.Unet model """
        super().__init__(in_channels=in_channels)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            decoder_channels=[256, 128, 64, 32, 64],
            in_channels=in_channels,
            classes=in_channels,
            aux_params=None,
            activation=None
        )
