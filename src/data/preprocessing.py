"""
Preprocessing for images
"""
import typing as tp
import numpy as np

import const
from src.data.base import Batch


def normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize image to [0, 1] range

    :param image: image
    :param eps: small number for avoiding division by zero
    :return: normalized image
    """
    image = image.astype(float)
    min_value = np.min(image)
    image -= min_value
    max_value = np.max(image) + eps
    image /= max_value
    return image

def normalize_batch(batch: Batch) -> Batch:
    """
    Normalize batch of images

    :param batch: batch of images
    :return: normalized batch
    """
    batch['image'] = normalize(batch['image'])
    return batch


def rotate_array(array_3d: np.ndarray) -> np.ndarray:
    """
    Rotate array by 90 degrees. Used to rotate images as masks.

    :param array_3d: array to rotate
    :return: rotated array
    """
    return np.rot90(array_3d, axes=(2, 1))

def crop_3d_mask_and_image(
        image: np.ndarray, mask: np.ndarray, crop_size: tp.Tuple[int, int, int]
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """ Crop 3d image and mask to crop_size """
    assert image.ndim == 3, '3d tensor must be provided'
    assert image.shape == mask.shape, 'image and mask must have the same shape'

    full_dim1, full_dim2, full_dim3 = image.shape
    dim1, dim2, dim3 = crop_size

    slice_min, w_min, h_min = np.random.randint(
        [dim1, dim2, dim3],
        [full_dim1 - dim1, full_dim2 - dim2, full_dim3 - dim3],
    )

    return (
        image[slice_min: slice_min + dim1, w_min: w_min + dim2, h_min: h_min + dim3],
        mask[slice_min: slice_min + dim1, w_min: w_min + dim2, h_min: h_min + dim3]
    )


def crop_3d_mask_and_image_batch(batch: Batch) -> Batch:
    """ Crop 3d image and mask to crop_size """
    batch['image'], batch['mask'] = crop_3d_mask_and_image(
        batch['image'], batch['mask'], crop_size=const.DEFAULT_CROP_SHAPE_IN_3D
    )
    return batch
