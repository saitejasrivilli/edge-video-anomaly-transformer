"""Anomaly operating threshold — derived from training data only, never test labels.

CLAUDE.md Phase 8 Section 14: choosing a threshold using test labels
would leak test information into an operating decision. Here, the
threshold is the given percentile of the TRAINING normal images' own
anomaly scores — e.g. the 95th percentile means "flag anything more
anomalous than all but the most unusual 5% of normal training images."
"""

from __future__ import annotations

import numpy as np


def derive_threshold_from_normal_scores(
    train_normal_scores: np.ndarray, percentile: float = 95.0
) -> float:
    """Threshold = the given percentile of ``train_normal_scores`` (training data only).

    Raises:
        ValueError: if ``percentile`` is not in (0, 100] or scores are empty.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if train_normal_scores.size == 0:
        raise ValueError("train_normal_scores must not be empty")

    return float(np.percentile(train_normal_scores, percentile))
