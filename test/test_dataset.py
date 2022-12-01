"""
Test src.dataset
"""

import typing as tp

import pytest
from torch.utils.data import DataLoader

from src.data import dataset


def _get_batch(dataloader: DataLoader) -> dataset.Batch:
    return next(iter(dataloader))


def test__types():
    """
    Assert typing.

    Can not check directly instance of dataset.Batch because
    "TypeError: TypedDict does not support instance and class checks"
    """
    standard_loaders = dataset.get_standard_dataloaders(batch_size=1, num_workers=1)
    assert isinstance(standard_loaders, dataset.Loaders)

    train_batch = _get_batch(standard_loaders.train)
    valid_batch = _get_batch(standard_loaders.valid)
    assert (keys := set(train_batch.keys())) == set(valid_batch.keys()), (
        f'Set of train batch keys {keys} != set of train batch keys {set(valid_batch.keys())}'
    )

    assert not set(dataset.Batch.__annotations__.keys()).symmetric_difference(keys), (
        f'Set of batch keys {keys} != set of dataset.Batch keys'
    )


def test__shapes():
    """
    Assert all images and masks have the same shapes.
    Unfortunately, images can not have different shapes because batching, but mask shape may differ from image shape.
    """
    dataloaders = dataset.get_standard_dataloaders()
    only_one_shape: tp.Tuple[int, ...] = _get_batch(dataloaders.train)['image'].shape
    for loader in iter(dataloaders):
        for batch in loader:
            assert batch['image'].shape == batch['mask'].shape == only_one_shape, (
                f"Shape does not match. For the image from first batch is {only_one_shape}. "
                f"For one of image is {batch['image'].shape}, mask {batch['mask'].shape}. "
                f"In loader {loader}."
            )


def test__dataloader_split():
    """ Raise error then have incorrect split for train/val """
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(split_lengths=(1, 0))

    dataset_len = sum(len(loader) for loader in iter(dataset.get_standard_dataloaders(batch_size=1)))
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(split_lengths=(dataset_len + 1, 0))
    with pytest.raises(ValueError):
        dataset.get_standard_dataloaders(split_lengths=(0, dataset_len + 1))
