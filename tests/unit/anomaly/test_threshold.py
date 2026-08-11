import numpy as np
import pytest

from evat.anomaly.threshold import derive_threshold_from_normal_scores


def test_threshold_is_percentile_of_scores():
    scores = np.arange(1, 101, dtype=np.float64)  # 1..100

    threshold = derive_threshold_from_normal_scores(scores, percentile=95.0)

    assert threshold == pytest.approx(np.percentile(scores, 95.0))


def test_threshold_rejects_invalid_percentile():
    scores = np.array([1.0, 2.0, 3.0])

    for bad in (0.0, -1.0, 100.1):
        with pytest.raises(ValueError, match="percentile"):
            derive_threshold_from_normal_scores(scores, percentile=bad)


def test_threshold_rejects_empty_scores():
    with pytest.raises(ValueError, match="empty"):
        derive_threshold_from_normal_scores(np.array([]))
