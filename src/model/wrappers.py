"""
Models for segmentation
"""

from src.model.base import BaseModel
from pytorch3dunet.unet3d.model import UNet2D, UNet3D, ResidualUNet3D
import segmentation_models_pytorch as smp


class Unet2DWrapper(BaseModel):
    """"""

    def __init__(self, in_channels: int):
        super().__init__(in_channels=in_channels)
        self.model = UNet2D(in_channels=in_channels, out_channels=in_channels, final_sigmoid=False)


class Unet3DWrapper(BaseModel):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.model = UNet3D(in_channels=in_channels, out_channels=out_channels, final_sigmoid=False)


class UnetSMPWrapper(BaseModel):
    """ Wrapper for smp.Unet """
    def __init__(self, in_channels: int, encoder_name: str = 'resnet34') -> None:
        """ Create model from segmentation_models_pytorch """
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
