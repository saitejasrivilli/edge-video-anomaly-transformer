import pytest
import torch

from evat.experiments.metrics import compute_classification_metrics


def test_perfect_predictions_give_full_scores():
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])

    metrics = compute_classification_metrics(preds, targets, num_classes=3)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0


def test_all_wrong_predictions_give_zero_scores():
    preds = torch.tensor([1, 0])
    targets = torch.tensor([0, 1])

    metrics = compute_classification_metrics(preds, targets, num_classes=2)

    assert metrics.accuracy == 0.0
    assert metrics.macro_f1 == 0.0


def test_imbalanced_classes_are_macro_averaged_not_accuracy_weighted():
    # 9 examples of class 0 (all correct), 1 example of class 1 (wrong).
    preds = torch.tensor([0] * 9 + [0])
    targets = torch.tensor([0] * 9 + [1])

    metrics = compute_classification_metrics(preds, targets, num_classes=2)

    assert metrics.accuracy == pytest.approx(0.9)
    assert metrics.macro_f1 < 0.9  # macro F1 penalized by missed minority class


def test_shape_mismatch_raises():
    preds = torch.tensor([0, 1])
    targets = torch.tensor([0, 1, 1])

    try:
        compute_classification_metrics(preds, targets, num_classes=2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
