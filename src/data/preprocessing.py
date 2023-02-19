"""
Preprocessing for images.
"""
import typing

import albumentations as albu
import numpy as np
import volumentations as vol
from albumentations.pytorch import ToTensorV2


def get_normalization_3d() -> typing.Callable[[np.ndarray], np.ndarray]:
    """Normalization for 3d images. Used for train and validation."""
    return vol.Normalize(range_norm=False, p=1.).apply


def rotate_array(array_3d: np.ndarray) -> np.ndarray:
    """
    Rotate array by 90 degrees. Used to rotate images as masks.

    :param array_3d: array to rotate
    :return: rotated array
    """
    return np.rot90(array_3d, axes=(2, 1))

def train_augmentation() -> albu.Compose:
    """Train augmentation. Don't use normalization here, because we normalize full 3d image."""
    return albu.Compose([
        albu.VerticalFlip(p=0.2),
        albu.OneOf([
            albu.ElasticTransform(p=1),
            albu.GridDistortion(num_steps=2, distort_limit=0.2, p=1),
        ], 0.3),
        albu.RandomSizedCrop(min_max_height=(350, 500), height=512, width=512, p=0.2),
        ToTensorV2(),
    ])

def valid_augmentation() -> albu.Compose:
    """Validation augmentation. Don't use normalization here, because we normalize full 3d image."""
    return albu.Compose([
        albu.Resize(512, 512),
        ToTensorV2(),
    ])
