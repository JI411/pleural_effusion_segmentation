"""
Test src.data.dataset
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import batching
from src.data import dataset


def _get_batch(dataloader: DataLoader) -> dataset.Batch:
    """ Get first batch from dataloader """
    return next(iter(dataloader))


@pytest.fixture(name='standard_loaders')
def fixture_standard_loaders():
    """ Get standard loaders with batch size=2 and num_workers=2 """
    return batching.get_standard_dataloaders(batch_size=2, num_workers=2)

def test__smoke(standard_loaders):
    """ Check that we can get batch from dataloaders """
    for loader in standard_loaders:
        for _ in loader:
            break


def test__types(standard_loaders):
    """
    Assert typing.

    Can not check directly instance of dataset.Batch because
    "TypeError: TypedDict does not support instance and class checks"
    """
    assert isinstance(standard_loaders, batching.Loaders)

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
        batching.get_standard_dataloaders(batch_size=1, split_lengths=(2, 0))

    dataset_len = sum(len(loader) for loader in iter(batching.get_standard_dataloaders(batch_size=1)))
    with pytest.raises(ValueError):
        batching.get_standard_dataloaders(batch_size=1, split_lengths=(dataset_len + 1, 0))
    with pytest.raises(ValueError):
        batching.get_standard_dataloaders(batch_size=1, split_lengths=(0, dataset_len + 1))


def test__normalization(standard_loaders):
    """ Check that normalization to [0, 1] works """
    zero, one, eps = torch.tensor(0.), torch.tensor(1.), 1e-6
    for loader in standard_loaders:
        for batch in loader:
            image = batch['image']
            assert (image_min := image.min()) >= zero - eps, f'Min image pixel value {image_min} less than 0'
            assert (image_max := image.max()) <= one + eps, f'Max image pixel value {image_max} greater than 1'
