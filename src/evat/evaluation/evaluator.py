"""Validation-loop evaluator: runs a model over a DataLoader and averages metrics."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from evat.evaluation.metrics import SegmentationMetrics, binarize, compute_segmentation_metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: str = "cpu"
) -> SegmentationMetrics:
    """Run ``model`` over every batch in ``dataloader`` and average pixel-level metrics."""
    model.eval()
    totals = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    for batch in dataloader:
        images = batch.image.to(device)
        targets = batch.mask.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = binarize(probs)

        batch_metrics = compute_segmentation_metrics(preds, targets)
        totals["iou"] += batch_metrics.iou
        totals["dice"] += batch_metrics.dice
        totals["precision"] += batch_metrics.precision
        totals["recall"] += batch_metrics.recall
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Cannot evaluate on an empty dataloader")

    return SegmentationMetrics(**{k: v / num_batches for k, v in totals.items()})
