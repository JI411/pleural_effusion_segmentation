"""
Test src.data.dataset.
"""

import pytest
from torch.utils.data import DataLoader

from src.data import batching
from src.data import dataset


def _get_batch(dataloader: DataLoader) -> dataset.Batch:
    """Get first batch from dataloader."""
    return next(iter(dataloader))


@pytest.fixture(name='standard_loaders')
def fixture_standard_loaders():
    """Get standard loaders with batch size=2 and num_workers=2."""
    return batching.get_standard_dataloaders(batch_size=2, num_workers=2)

def test__smoke(standard_loaders):
    """Check that we can get batch from dataloaders."""
    for loader in standard_loaders:
        for _ in loader:
            break


def test__shape(standard_loaders):
    """Check that all batches have same shape."""
    first_sample = _get_batch(standard_loaders[0])
    shape = {'image': first_sample['image'].shape, 'mask': first_sample['mask'].shape}
    for loader in standard_loaders:
        for batch in loader:
            assert batch['image'].shape[1:] == shape['image'][1:], (
                f'Image shape {batch["image"].shape} != {shape["image"]}'
            )
            assert batch['mask'].shape[1:] == shape['mask'][1:], (
                f'Mask shape {batch["mask"].shape} != {shape["mask"]}'
            )


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
