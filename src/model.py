"""
Models for segmentation
"""
import torch
import segmentation_models_pytorch as smp

class UnetSMPWrapper(torch.nn.Module):  # pylint: disable=too-few-public-methods
    """ Wrapper for smp.Unet """

    def __init__(self, in_channels: int, encoder_name: str = 'resnet34') -> None:
        """ Create model from segmentation_models_pytorch """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            decoder_channels=[256, 128, 64, 32, 64],
            in_channels=in_channels,
            classes=in_channels,
            aux_params=None,
            activation=None
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """ Run model on image """
        result = self.model(image)
        return torch.sigmoid(result)
