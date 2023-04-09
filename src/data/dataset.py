"""
Dataset and Dataloader classes.
"""
import typing as tp

import albumentations as albu
import numpy as np

import const
from src.data.base import BaseDataset, Sample, load_sample


class PleuralEffusionDataset3D(BaseDataset):
    """Pleural Effusion Dataset for 3D training."""

    def __init__(
            self,
            data_dir: const.PathType,
            augmentation: albu.Compose = None,
            use_cache: bool = True
    ) -> None:
        """
        Create dataset class.

        :param data_dir: dir with data in Supervisely format; default const.IMAGES_DIR
        :param augmentation: augmentation function; default None
        :param use_cache: if True, load images and masks in memory
        """
        super().__init__(data_dir, augmentation)
        self.cache = self._get_samples_cache() if use_cache else None

    def _get_samples_cache(self) -> tp.Tuple[Sample]:
        """Read images and masks from disk, preprocess they and cache in memory."""
        return tuple(load_sample(paths) for paths in self.dataset)

    def __len__(self) -> int:
        """Len of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Sample:
        """Get (1, D, H, W)` lung image and mask for it."""
        if self.cache is not None:
            sample = self.cache[idx]
        else:
            sample = load_sample(self.dataset[idx])

        image, mask = sample['image'], sample['mask']
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        image, mask = np.transpose(image, (3, 2, 0, 1)), np.transpose(mask, (3, 2, 0, 1))
        image, mask = image.astype(np.float32), mask.astype(np.int)
        return Sample(image=image, mask=mask)
