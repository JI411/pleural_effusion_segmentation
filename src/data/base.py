import typing as tp
from pathlib import Path

import torch

import numpy as np
from torch.utils.data import DataLoader, Dataset

import const
from src.data import read_data


class Batch(tp.TypedDict):
    """ Contains batch from dataset """
    image: tp.Union[np.ndarray, torch.Tensor]
    mask: tp.Union[np.ndarray, torch.Tensor]

class Loaders(tp.NamedTuple):
    """ Contains split dataloaders """
    train: DataLoader
    valid: DataLoader


class PleuralEffusionDataset(Dataset):
    """ Pleural Effusion Dataset """

    fill_value_mask = 0
    fill_value_image = -1024

    def __init__(
            self,
            images_dir: const.PathType = const.IMAGES_DIR,
            masks_dir: const.PathType = const.MASKS_DIR,
    ) -> None:
        """
        Create dataset class
        :param images_dir: dir with dirs with .dcm images; default const.IMAGES_DIR
        :param masks_dir: dir with dirs with .nii.gz masks; default const.MASKS_DIR
        """
        self.image_dir_paths: tp.List[Path] = sorted([p for p in Path(images_dir).glob('*') if p.is_dir()])
        self.masks_dir_paths: tp.List[Path] = sorted([p for p in Path(masks_dir).glob('*') if p.is_dir()])
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
        raise NotImplementedError
