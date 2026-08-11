import numpy as np
import pytest

from evat.anomaly.metrics import pr_auc_score, roc_auc_score


def test_roc_auc_perfect_separation_is_one():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    assert roc_auc_score(scores, labels) == pytest.approx(1.0)


def test_roc_auc_inverted_separation_is_zero():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])

    assert roc_auc_score(scores, labels) == pytest.approx(0.0)


def test_roc_auc_random_is_about_half():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=2000)
    labels = rng.integers(0, 2, size=2000)

    auc = roc_auc_score(scores, labels)

    assert 0.4 < auc < 0.6


def test_roc_auc_handles_ties():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])

    assert roc_auc_score(scores, labels) == pytest.approx(0.5)


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        roc_auc_score(np.array([0.1, 0.2]), np.array([0, 0]))


def test_pr_auc_perfect_separation_is_one():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    assert pr_auc_score(scores, labels) == pytest.approx(1.0)


def test_pr_auc_requires_at_least_one_positive():
    with pytest.raises(ValueError, match="one positive"):
        pr_auc_score(np.array([0.1, 0.2]), np.array([0, 0]))
