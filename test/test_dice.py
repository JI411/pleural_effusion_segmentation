"""
Test src.dice
"""

import torch

import const
from src import dice


def inverse_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """ Inverse function for sigmoid """
    return -torch.log((1 / (logits + 1e-8)) - 1)


class TestDiceLoss:
    """ Contains test for dice loss values for different answers """

    batch_mask = torch.rand(
        2, 100, 512, 512,
        device='cpu',
        dtype=torch.float,
        generator=torch.Generator().manual_seed(const.SEED),
        requires_grad=False
    ) < 0.5

    loss = dice.BinaryDiceLoss()

    def test_best_predict(self):
        """ If (pred == mask) loss must be equal to zero """
        score = self.loss(raw_logits=inverse_sigmoid(self.batch_mask), mask=self.batch_mask)
        assert torch.isclose(score, torch.tensor(0, dtype=torch.float)), score

    def test_worst_predict(self):
        """ If (pred != mask) in every point loss must be equal to one """
        inverse_mask = self.batch_mask.clone().bool()
        inverse_mask = (~inverse_mask).float()
        score = self.loss(raw_logits=inverse_sigmoid(inverse_mask), mask=self.batch_mask)
        assert torch.isclose(score, torch.tensor(1, dtype=torch.float))
