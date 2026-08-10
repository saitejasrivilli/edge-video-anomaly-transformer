"""Segmentation loss.

Model output: raw logits, shape [B, 1, H, W] (binary foreground/background
baseline — see docs/architecture.md for the object-representation
decision). Target: binary mask, shape [B, 1, H, W], values in {0, 1}.

Loss = Binary Cross-Entropy (on logits, via ``BCEWithLogitsLoss`` for
numerical stability) + Dice loss (on sigmoid-activated probabilities).
BCE alone is well-understood and stable early in training but is
insensitive to class imbalance (foreground objects are usually a small
fraction of pixels); Dice directly targets mask overlap and compensates
for that imbalance. Combining both is a standard, explainable choice for
a first baseline — this is not a tuned or exotic loss.
"""

from __future__ import annotations

import torch
from torch import nn


def dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. ``probs`` are sigmoid-activated predictions in [0, 1]."""
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE-with-logits + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = dice_loss(probs, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
