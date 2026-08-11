"""Threshold-free anomaly-detection metrics: ROC-AUC and PR-AUC.

No scikit-learn dependency added (consistent with
``evat.experiments.metrics``): both are computed directly.

- ROC-AUC via the Mann-Whitney U statistic:
  ``AUC = (sum of ranks of the positive class - n_pos*(n_pos+1)/2) / (n_pos * n_neg)``.
  This is an exact, closed-form equivalent of the area under the ROC
  curve for binary labels — not an approximation.
- PR-AUC via trapezoidal integration over the precision-recall curve
  swept across every distinct score value as a threshold.

Both metrics are threshold-free (they summarize ranking quality across
all thresholds), consistent with CLAUDE.md Phase 8 Section 14 (do not
tune a threshold against test labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AnomalyMetrics:
    roc_auc: float
    pr_auc: float


def roc_auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """``scores``: higher = more anomalous. ``labels``: 1 = anomalous, 0 = normal.

    Raises:
        ValueError: if only one class is present (AUC undefined).
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires both classes to be present")

    ranks = _rank(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def pr_auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, swept over every distinct score threshold."""
    if (labels == 1).sum() == 0:
        raise ValueError("pr_auc_score requires at least one positive (anomalous) example")

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp_cumulative = np.cumsum(sorted_labels)
    fp_cumulative = np.cumsum(1 - sorted_labels)
    total_positive = sorted_labels.sum()

    precision = tp_cumulative / (tp_cumulative + fp_cumulative)
    recall = tp_cumulative / total_positive

    # Prepend the (recall=0, precision=1) point for correct integration.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    return float(np.trapezoid(precision, recall))


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-indexed), correctly handling ties — required for exact AUC."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    i = 0
    while i < len(sorted_values):
        j = i
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        average_rank = (i + 1 + j) / 2.0  # 1-indexed average over the tied block [i, j)
        ranks[order[i:j]] = average_rank
        i = j

    return ranks
