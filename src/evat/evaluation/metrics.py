"""Pixel-level segmentation metrics.

These are PIXEL-LEVEL metrics computed over binarized predictions vs.
binary ground truth for a single frame (or a batch of frames, averaged).
They are not object-level metrics (e.g. per-instance matching, J&F as used
in the official DAVIS/YouTube-VOS benchmarks) — this project does not
claim video-object-segmentation benchmark performance, only frame-level
foreground/background overlap quality for its own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    """Pixel-level metrics for one evaluation pass."""

    iou: float
    dice: float
    precision: float
    recall: float


def binarize(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Threshold sigmoid probabilities into a binary {0, 1} mask."""
    return (probs >= threshold).float()


def compute_segmentation_metrics(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6
) -> SegmentationMetrics:
    """Compute IoU, Dice, precision, recall between binary masks.

    Both inputs must already be binary ({0, 1}) tensors of the same shape.
    Metrics are computed over all elements (batch, channel, and spatial
    dims flattened together).
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    pred = pred_mask.flatten().float()
    gt = gt_mask.flatten().float()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    intersection = tp
    union = pred.sum() + gt.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return SegmentationMetrics(
        iou=iou.item(),
        dice=dice.item(),
        precision=precision.item(),
        recall=recall.item(),
    )
