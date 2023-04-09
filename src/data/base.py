"""
Batching and data loading.
"""

import typing as tp

import albumentations as albu
import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset

import const


class SamplePath(tp.TypedDict):
    """Contains batch from dataset."""
    image: const.PathType
    mask: const.PathType

class Sample(tp.TypedDict):
    """Contains batch from dataset."""
    image: tp.Union[np.ndarray, torch.Tensor]
    mask: tp.Union[np.ndarray, torch.Tensor]


def load_array(path: const.PathType) -> np.ndarray:
    """
    Read image or mask from .nrrd file

    :param path: path to .nrrd file
    :return: 3d image
    """
    data, _ = nrrd.read(path)
    return data

def load_sample(paths: SamplePath) -> Sample:
    """Load image and mask from paths."""
    image, mask = load_array(paths['image']), load_array(paths['mask'])
    image, mask = np.expand_dims(image, axis=3), np.expand_dims(mask, axis=3)
    image, mask = image.astype(np.float32), mask.astype(np.int)
    return Sample(image=image, mask=mask)


class BaseDataset(Dataset):
    """Pleural Effusion Dataset."""

    def __init__(
            self,
            data_dir: const.PathType,
            augmentation: albu.Compose = None,
    ) -> None:
        """
        Create dataset class.

        :param data_dir: dir with data in Supervisely format; default const.IMAGES_DIR
        :param augmentation: augmentation function; default None
        """
        self.augmentation = augmentation
        self.data_dir = data_dir
        self.dataset = sorted(list(self._get_paths()), key=str)

    def _get_paths(self) -> tp.Iterator[SamplePath]:
        """Get paths to images and masks."""
        images_dir = self.data_dir / const.DatasetPathConfig.images_dir_name
        masks_dir = self.data_dir / const.DatasetPathConfig.masks_dir_name
        for image_path in images_dir.glob('*.nrrd'):
            mask_path = masks_dir / image_path.name / const.DatasetPathConfig.mask_file_name
            yield SamplePath(image=image_path, mask=mask_path)

    def __getitem__(self, idx: int) -> Sample:
        """Get lung image and mask for it."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Len of dataset."""
        raise NotImplementedError
