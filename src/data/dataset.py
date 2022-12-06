"""
Dataset and Dataloader classes
"""

import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import const
from src.data import read_data
from src.data.base import PleuralEffusionDataset, Batch, Loaders


class PleuralEffusionDataset2D(PleuralEffusionDataset):
    """ Pleural Effusion Dataset """

    def __init__(self, images_dir: const.PathType = const.IMAGES_DIR, masks_dir: const.PathType = const.MASKS_DIR,
                 num_channels: tp.Optional[int] = None) -> None:
        """
        Create dataset class
        :param images_dir: dir with dirs with .dcm images; default const.IMAGES_DIR
        :param masks_dir: dir with dirs with .nii.gz masks; default const.MASKS_DIR
        :param num_channels: num channels in one sample, set for all images; default use max channels in dataset
        """

        super().__init__(images_dir, masks_dir)
        self.num_channels: int = num_channels or max(x.shape[0] for x in read_data.load_dicom_recursive(images_dir))

    def __getitem__(self, idx: int) -> Batch:
        """ Get lung image and mask for it """
        image = list(read_data.load_dicom_recursive(self.image_dir_paths[idx]))[0]
        mask = read_data.load_mask_from_dir(self.masks_dir_paths[idx])

        if (shape := image.shape)[0] < self.num_channels:
            empty_layers_shape = (self.num_channels - shape[0], *shape[1:3])

            image = np.concatenate([
                image, np.full(shape=empty_layers_shape, fill_value=self.fill_value_image)
            ])
            mask = np.concatenate([
                mask, np.full(shape=empty_layers_shape, fill_value=self.fill_value_mask)
            ])
        elif shape[0] > self.num_channels:
            image = image[:self.num_channels]
            mask = mask[:self.num_channels]

        return Batch(image=image.astype('float32'), mask=mask.astype(int))

class PleuralEffusionDataset3D(PleuralEffusionDataset):
    """ Pleural Effusion Dataset """

    def __getitem__(self, idx: int) -> Batch:
        """ Get lung image and mask for it """
        image = list(read_data.load_dicom_recursive(self.image_dir_paths[idx]))[0]
        mask = read_data.load_mask_from_dir(self.masks_dir_paths[idx])
        return Batch(image=image[None].astype('float32'), mask=mask[None].astype(int))


def get_standard_dataloaders(
        batch_size: int,
        num_workers: int = const.DEFAULT_NUM_WORKERS,
        split_lengths: tp.Optional[tp.Tuple[int, int]] = None,
        **kwargs
) -> Loaders:
    """
    Get dataloaders to current dataset
    :param batch_size: how many samples per batch to load
    :param num_workers: how many subprocesses to use for data loading. 0 => no multiprocessing
    :param split_lengths: lengths of splits to be produced, first for train, second for valid
    :param kwargs: params for PleuralEffusionDataset
    :return: (train_dataloader, valid_dataloader)
    """
    if split_lengths is None:
        full_data_len = len(PleuralEffusionDataset3D(**kwargs))
        num_valid_samples = int(const.DEFAULT_VALID_FRACTION * full_data_len)
        split_lengths = (full_data_len - num_valid_samples, num_valid_samples)

    train, valid = random_split(
        PleuralEffusionDataset3D(**kwargs), split_lengths, generator=torch.Generator().manual_seed(const.SEED)
    )
    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return Loaders(train=train, valid=valid)
