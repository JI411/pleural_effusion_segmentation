"""
Test src.wrappers.
"""
from typing import Tuple

import pytest
import torch

from src.model import wrappers
from src.model.base import BaseModelWrapper


@pytest.mark.parametrize(
    "input_shape,out_shape,net", [
        ((2, 1, 32, 64, 64), (2, 1, 32, 64, 64), wrappers.UNet3DWrapper()),
    ]
)
def test__wrapper(input_shape: Tuple[int, ...], out_shape: Tuple[int, ...], net: BaseModelWrapper):
    """
    Test wrappers for segmentation models. Run model on random tensor and check output shape.

    :param net: wrapper for model
    :return:
    """
    net.train()
    input_tensor = torch.rand(*input_shape)
    ideal_out = torch.rand(*out_shape)
    out = net.forward(input_tensor)
    if ideal_out.shape != out.shape:
        raise ValueError(
            f"Expected output shape {ideal_out.shape}, but got {out.shape}"
        )
