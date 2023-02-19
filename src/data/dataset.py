"""
Dataset and Dataloader classes.
"""
import typing as tp

import albumentations as albu
import numpy as np

import const
from src.data import read_data, preprocessing
from src.data.base import BaseDataset, Batch

T = tp.TypeVar('T')

def split_channels(array: np.ndarray) -> tp.List[np.ndarray]:
    """Split array to channels (axis=0)."""
    return np.split(array, array.shape[0], axis=0)

def flatten(list_of_lists: tp.List[tp.List[T]]) -> tp.List[T]:
    """Convert list of lists to list."""
    return [item for sublist in list_of_lists for item in sublist]

class PleuralEffusionDataset2D(BaseDataset):
    """Pleural Effusion Dataset for 2D training."""

    def __init__(
            self,
            images_dir: const.PathType,
            masks_dir: const.PathType,
            augmentation: albu.Compose = None,
    ) -> None:
        """
        Create dataset class.

        Images read and caching in memory, because reading from disk is too slow for 2D variant.
        :param images_dir: dir with dirs with .dcm images; default const.IMAGES_DIR
        :param masks_dir: dir with dirs with .nii.gz masks; default const.MASKS_DIR
        :param augmentation: augmentation function; default None
        """
        super().__init__(images_dir, masks_dir, augmentation)
        self.normalization = preprocessing.get_normalization_3d()
        self.images = self.get_images_cache()
        self.masks = self.get_masks_cache()

    def get_images_cache(self) -> tp.List[np.ndarray]:
        """Read images from disk, preprocess they and cache in memory."""
        images = [list(read_data.load_dicom_recursive(p))[0] for p in self.image_dir_paths]
        images = [preprocessing.rotate_array(img) for img in images]
        images = [self.normalization(img) for img in images]
        images = [split_channels(img) for img in images]
        images = flatten(images)
        return [np.moveaxis(image, 0, -1) for image in images]

    def get_masks_cache(self) -> tp.List[np.ndarray]:
        """Read masks from disk and cache in memory."""
        masks = [read_data.load_mask_from_dir(p) for p in self.masks_dir_paths]
        masks = [split_channels(mask) for mask in masks]
        masks = flatten(masks)
        return [np.moveaxis(mask, 0, -1) for mask in masks]

    def __len__(self) -> int:
        """Len of dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Batch:
        """Get (1 x W x H ) lung image and mask for it."""
        image = self.images[idx].astype('float32')
        mask = self.masks[idx].astype(int)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return Batch(image=image, mask=mask)
