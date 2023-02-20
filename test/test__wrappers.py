"""
Test src.wrappers.
"""
import typing as tp
from functools import partial

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset

import const
from src.model import wrappers
from src.model.base import BaseModel


def create_sphere(shape: tp.Sequence, radius: int, position: tp.Sequence, eps=1e-5) -> np.ndarray:
    """
    Generate an n-dimensional sphere.

    :param shape: shape of the sphere, len(shape) == n-dims
    :param radius: radius of the sphere
    :param position: center of the sphere
    :param eps: small epsilon for numerical stability
    :return: sphere
    """
    assert len(position) == len(shape)
    semi_sizes = [radius for _ in shape]
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semi_size in zip(position, semi_sizes):
        arr += (x_i / (semi_size + eps)) ** 2
    return (arr <= 1.0).astype('float32')


def _random_sphere_dataloader(shape: tp.Sequence[int]) -> DataLoader:
    """
    Create dataloader with one batch of random spheres.

    :param shape: shape of batch, first value is batch size
    :return: dataloader
    """
    shape = list(shape)
    batch_size, shape = shape[0], shape[1:]
    rng = np.random.default_rng(const.SEED)
    create_sphere_with_fixed_shape = partial(create_sphere, shape=shape, position=[int(dim / 2) for dim in shape])
    sphere = torch.tensor([
        create_sphere_with_fixed_shape(radius=rng.integers(low=max(shape) // 5, high=max(shape) // 2))
        for _ in range(batch_size)
    ])
    dataset = TensorDataset(sphere, sphere)
    return DataLoader(dataset, batch_size=batch_size)

@pytest.mark.parametrize(
    "input_shape,num_epoch,net,max_allowed_loss", [
        ((4, 1, 128, 128), 20, wrappers.UnetSMP2DWrapper(), 0.025),
    ]
)
def test__wrapper(input_shape: tp.Sequence[int], num_epoch: int, net: BaseModel, max_allowed_loss: float):
    """
    Test wrappers for segmentation models on simple task.

    :param input_shape: shape of input
    :param num_epoch: number of epochs for training model before test
    :param net: wrapper for model
    :param max_allowed_loss:  max allowed loss on simple train dataset after training
    :return:
    """
    seed_everything(const.SEED, workers=True)
    net.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    loader = _random_sphere_dataloader(shape=input_shape)
    for _ in range(num_epoch):
        loss = torch.tensor(0.)
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = net(inputs).flatten()
            labels = labels.flatten()
            loss += criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    loss = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = net(inputs).flatten()
            loss.append(criterion(outputs, labels.flatten()).item())
    assert np.mean(loss) < max_allowed_loss, f'loss must be lower than {max_allowed_loss}, but have {loss=}'
