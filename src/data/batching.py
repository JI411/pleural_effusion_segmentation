"""
Preprocess dataset before training
"""

import typing as tp
from functools import partial

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import collate, default_collate_fn_map

import const
from src.data.base import Loaders
from src.data.dataset import PleuralEffusionDataset3D


def collate_padded_tensor_fn(batch, *, collate_fn_map=None):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.unsqueeze(1)

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
    padding_collate_fn_map = default_collate_fn_map.copy()
    padding_collate_fn_map.update({torch.Tensor: collate_padded_tensor_fn})
    train = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate, collate_fn_map=padding_collate_fn_map)
    )
    valid = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate, collate_fn_map=padding_collate_fn_map)
    )
    return Loaders(train=train, valid=valid)
