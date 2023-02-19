"""
Preprocess dataset before training.
"""

import typing as tp
from torch.utils.data import DataLoader

import const
from src.data.base import Loaders, BaseDataset
from src.data import dataset, preprocessing

def get_standard_dataloaders(
        batch_size: int,
        num_workers: int = const.DEFAULT_NUM_WORKERS,
        dataset_class: tp.Type[BaseDataset] = dataset.PleuralEffusionDataset2D,
) -> Loaders:
    """
    Get dataloaders to dataset.

    :param batch_size: how many samples per batch to load
    :param num_workers: how many subprocesses to use for data loading. 0 => no multiprocessing
    :param dataset_class: dataset class for create loaders
    :return: (train_dataloader, valid_dataloader)
    """
    train = dataset_class(
        images_dir=const.DatasetPathConfig.train_images,
        masks_dir=const.DatasetPathConfig.train_masks,
        augmentation=preprocessing.train_augmentation(),
    )
    valid = dataset_class(
        images_dir=const.DatasetPathConfig.valid_images,
        masks_dir=const.DatasetPathConfig.valid_masks,
        augmentation=preprocessing.valid_augmentation(),
    )
    train = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return Loaders(train=train, valid=valid)
