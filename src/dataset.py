"""
Dataset and Dataloader classes
"""

import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import const
from src import read_data


class Batch(tp.TypedDict):
    """ Contains batch from dataset """
    image: np.ndarray
    mask: np.ndarray

class PleuralEffusionDataset(Dataset):
    """Pleural Effusion Dataset"""

    fill_value_mask = 0
    fill_value_image = -1024

    def __init__(
            self,
            images_dir: tp.Union[Path, str] = const.IMAGES_DIR,
            masks_dir: tp.Union[Path, str] = const.MASKS_DIR,
            num_channels: tp.Optional[int] = None
    ) -> None:
        """
        Create dataset class
        :param images_dir: dir with dirs with .dcm images; default const.IMAGES_DIR
        :param masks_dir: dir with dirs with .nii.gz masks; default const.MASKS_DIR
        :param num_channels: num channels in one sample, set for all images; default use max channels in dataset
        """
        self.image_dir_paths: tp.List[Path] = sorted([p for p in images_dir.glob('*') if p.is_dir()])
        self.masks_dir_paths: tp.List[Path] = sorted([p for p in masks_dir.glob('*') if p.is_dir()])
        self.num_channels: int = num_channels or max(x.shape[0] for x in read_data.load_dicom_recursive(images_dir))
        self._check_paths()

    def _check_paths(self) -> None:
        """
        Check paths to images and masks
        :raise ValueError: if names in image_sir_names is different from names in masks_dir_paths
        :return:
        """
        if [p.name for p in self.image_dir_paths] != [p.name for p in self.masks_dir_paths]:
            raise ValueError('Dataset object names does not match!')

    def __len__(self) -> int:
        """ Len of dataset """
        return len(self.image_dir_paths)

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

        return Batch(image=image, mask=mask)


def get_standard_dataloaders(
        batch_size: int = 2, num_workers: int = 2, split_lengths: tp.Sequence[int] = (7, 3)
) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Get dataloaders to current dataset
    :param batch_size: how many samples per batch to load
    :param num_workers: how many subprocesses to use for data loading. 0 => no multiprocessing
    :param split_lengths: lengths of splits to be produced
    :return: (train_dataloader, valid_dataloader)
    """
    train, valid = random_split(
        PleuralEffusionDataset(), split_lengths, generator=torch.Generator().manual_seed(const.SEED)
    )
    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, valid
