"""
Compute Dice loss.
"""

import torch
from torch.nn.functional import logsigmoid
from torch.nn.modules.loss import _Loss

class BinaryDiceLoss(_Loss):
    """
    Dice loss for binary mask.
    Reference: https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html#DiceLoss
    """

    def __init__(self, smooth: float = 0.0, eps: float = 1e-7) -> None:
        """
        Init class.

        :param smooth: smoothness constant for dice coefficient
        :param eps: small value to avoid zero division
        :return:
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, raw_logits: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Compute Dice loss.

        :param raw_logits: network output without sigmoid
        :param mask: ground true
        :return: dice loss
        """
        batch_size = mask.size(0)
        dims = (0, 2)
        mask = mask.bool()
        mask = mask.view(batch_size, 1, -1)
        mask_pred = raw_logits.view(batch_size, 1, -1)
        mask_pred = logsigmoid(mask_pred).exp()

        intersection = torch.sum(mask_pred * mask, dim=dims)
        cardinality = torch.sum(mask_pred + mask, dim=dims)
        score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        return 1 - score
