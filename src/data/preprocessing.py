"""
Preprocessing for images
"""
import numpy as np

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


def rotate_array(array_3d: np.ndarray) -> np.ndarray:
    """
    Rotate array by 90 degrees. Used to rotate images as masks.

    :param array_3d: array to rotate
    :return: rotated array
    """
    return np.rot90(array_3d, axes=(2, 1))
