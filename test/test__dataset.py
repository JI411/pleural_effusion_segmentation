"""
Test src.data.dataset
"""

import typing as tp

import pytest
from torch.utils.data import DataLoader

from src.data import dataset


def _get_batch(dataloader: DataLoader) -> dataset.Batch:
    return next(iter(dataloader))


@pytest.fixture()
def standard_loaders():
    return dataset.get_standard_dataloaders(batch_size=2, num_workers=2)

def test__smoke(standard_loaders):
    for loader in standard_loaders:
        for _ in loader:
            break


def test__types(standard_loaders):
    """
    Assert typing.

    Can not check directly instance of dataset.Batch because
    "TypeError: TypedDict does not support instance and class checks"
    """
    assert isinstance(standard_loaders, dataset.Loaders)

    train_batch = _get_batch(standard_loaders.train)
    valid_batch = _get_batch(standard_loaders.valid)
    assert (keys := set(train_batch.keys())) == set(valid_batch.keys()), (
        f'Set of train batch keys {keys} != set of train batch keys {set(valid_batch.keys())}'
    )

    assert not set(dataset.Batch.__annotations__.keys()).symmetric_difference(keys), (
        f'Set of batch keys {keys} != set of dataset.Batch keys'
    )

def test__dataloader_split():
    """ Raise error then have incorrect split for train/val """
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(batch_size=1, split_lengths=(1, 0))

    dataset_len = sum(len(loader) for loader in iter(dataset.get_standard_dataloaders(batch_size=1)))
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(batch_size=1, split_lengths=(dataset_len + 1, 0))
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(batch_size=1, split_lengths=(0, dataset_len + 1))
