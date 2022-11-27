"""
Compute Dice loss
Reference: https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html#DiceLoss
"""

import torch
from torch.nn.functional import logsigmoid
from torch.nn.modules.loss import _Loss

class BinaryDiceLoss(_Loss):

    def __init__(self, smooth: float = 0.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, raw_logits: torch.Tensor, mask: torch.Tensor) -> float:
        bs = mask.size(0)
        dims = (0, 2)
        mask = mask.view(bs, 1, -1)
        mask_pred = raw_logits.view(bs, 1, -1)
        mask_pred = logsigmoid(mask_pred).exp()
        # mask_pred = torch.sigmoid(mask_pred)

        intersection = torch.sum(mask_pred * mask, dim=dims)
        cardinality = torch.sum(mask_pred + mask, dim=dims)
        score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        return 1 - score

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
