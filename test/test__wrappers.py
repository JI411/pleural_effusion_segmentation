"""
Test src.wrappers
"""
import typing as tp

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import wrappers
from src.model.base import BaseModel


def _easy_dataloader(shape: tp.Sequence[int]) -> DataLoader:
    """
    Create dataloader with one batch of ones
    :param shape: shape of batch
    :return: dataloader
    """
    rand = (torch.rand(*shape) > 0.5).float()
    dataset = TensorDataset(rand, rand)
    return DataLoader(dataset, batch_size=1)

@pytest.mark.parametrize(
    "input_shape,num_epoch,net,max_allowed_loss", [
        ((8, 1, 16, 32, 32), 10, wrappers.Unet3DWrapper(), 0.06),
    ]
)
def test__wrapper(input_shape: tp.Sequence[int], num_epoch: int, net: BaseModel, max_allowed_loss: float):
    """
    Test wrappers for segmentation models on simple task

    :param input_shape: shape of input
    :param num_epoch: number of epochs for training model before test
    :param net: wrapper for model
    :param max_allowed_loss:  max allowed loss on simple train dataset after training
    :return:
    """
    net.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loader = _easy_dataloader(shape=input_shape)
    for _ in range(num_epoch):
        for inputs, labels in loader:
            optimizer.zero_grad()

            outputs = net(inputs).flatten()
            labels = labels.flatten()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    loss = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = net(inputs)
            loss.append(criterion(outputs, labels).item())
    assert np.mean(loss) < max_allowed_loss, f'loss must be lower than {max_allowed_loss}, but have {loss=}'
