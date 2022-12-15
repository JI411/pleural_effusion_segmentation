"""
Preprocess dataset before training
"""

import typing as tp
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils import collate

import const
from src.data.base import Loaders, BaseDataset
from src.data import dataset


def pad_channels_to_max(arrays: tp.List[np.ndarray], value: int) -> tp.List[np.ndarray]:
    """ Pad numpy arrays channels to max size """
    shapes = np.array([x.shape for x in arrays])
    max_size = shapes.max(axis=0, keepdims=True, initial=-1)  # pylint: disable=unexpected-keyword-arg
    margin = (max_size - shapes)
    pad_before = margin // 2
    pad_after = margin - pad_before
    pad = np.stack([pad_before, pad_after], axis=2)
    return [np.pad(x, w, mode='constant', constant_values=value) for x, w in zip(arrays, pad)]

def pad_collate_numpy_array_fn(batch, *, collate_fn_map=None):
    """ Pad numpy arrays channels to max size and call default collate to create tensor """
    elem = batch[0]
    # array of string classes and object
    if collate.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(collate.default_collate_err_msg_format.format(elem.dtype))

    batch = pad_channels_to_max(batch, value=0)
    return collate.collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)

def get_standard_dataloaders(
        batch_size: int,
        num_workers: int = const.DEFAULT_NUM_WORKERS,
        split_lengths: tp.Optional[tp.Tuple[int, int]] = None,
        dataset_class: tp.Type[BaseDataset] = dataset.PleuralEffusionDataset2D,
        **kwargs
) -> Loaders:
    """
    Get dataloaders to dataset

    :param batch_size: how many samples per batch to load
    :param num_workers: how many subprocesses to use for data loading. 0 => no multiprocessing
    :param split_lengths: lengths of splits to be produced, first for train, second for valid
    :param dataset_class: dataset class for create loaders
    :param kwargs: params for BaseDataset
    :return: (train_dataloader, valid_dataloader)
    """
    if split_lengths is None:
        full_data_len = len(dataset_class(**kwargs))
        num_valid_samples = int(const.DEFAULT_VALID_FRACTION * full_data_len)
        split_lengths = (full_data_len - num_valid_samples, num_valid_samples)

    train, valid = random_split(
        dataset_class(**kwargs), split_lengths, generator=torch.Generator().manual_seed(const.SEED)
    )
    padding_collate_fn_map = collate.default_collate_fn_map.copy()
    padding_collate_fn_map.update({np.ndarray: pad_collate_numpy_array_fn})
    train = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate.collate, collate_fn_map=padding_collate_fn_map)
    )
    valid = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate.collate, collate_fn_map=padding_collate_fn_map)
    )
    return Loaders(train=train, valid=valid)
