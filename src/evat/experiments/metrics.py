"""Classification metrics for the Phase 7 category task.

No scikit-learn dependency added: computed directly from per-class
confusion counts, kept dependency-minimal per CLAUDE.md Section 27.
Macro-averaged (not micro/accuracy-weighted) because YouTube-VOS
categories are imbalanced — see docs/task_definition.md "Evaluation".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float


def compute_classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> ClassificationMetrics:
    """``predictions``, ``targets``: ``[N]`` integer class indices."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    accuracy = (predictions == targets).float().mean().item() if targets.numel() else 0.0

    precisions, recalls, f1s = [], [], []
    for class_index in range(num_classes):
        pred_positive = predictions == class_index
        actual_positive = targets == class_index

        tp = (pred_positive & actual_positive).sum().item()
        fp = (pred_positive & ~actual_positive).sum().item()
        fn = (~pred_positive & actual_positive).sum().item()

        if actual_positive.sum().item() == 0:
            continue  # class absent from this batch/split; excluded from the macro average

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )
