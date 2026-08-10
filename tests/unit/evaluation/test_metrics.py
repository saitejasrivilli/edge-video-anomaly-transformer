import pytest
import torch

from evat.evaluation.metrics import binarize, compute_segmentation_metrics


def test_perfect_prediction_gives_iou_and_dice_near_one():
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).unsqueeze(0)

    metrics = compute_segmentation_metrics(mask, mask)

    assert metrics.iou == pytest.approx(1.0, abs=1e-4)
    assert metrics.dice == pytest.approx(1.0, abs=1e-4)
    assert metrics.precision == pytest.approx(1.0, abs=1e-4)
    assert metrics.recall == pytest.approx(1.0, abs=1e-4)


def test_disjoint_prediction_gives_zero_overlap():
    pred = torch.tensor([[1.0, 0.0], [0.0, 0.0]]).unsqueeze(0)
    gt = torch.tensor([[0.0, 0.0], [0.0, 1.0]]).unsqueeze(0)

    metrics = compute_segmentation_metrics(pred, gt)

    assert metrics.iou == pytest.approx(0.0, abs=1e-4)
    assert metrics.precision == pytest.approx(0.0, abs=1e-4)
    assert metrics.recall == pytest.approx(0.0, abs=1e-4)


def test_shape_mismatch_raises():
    a = torch.zeros(1, 2, 2)
    b = torch.zeros(1, 3, 3)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_segmentation_metrics(a, b)


def test_binarize_thresholds_correctly():
    probs = torch.tensor([0.1, 0.5, 0.6, 0.9])

    result = binarize(probs, threshold=0.5)

    assert result.tolist() == [0.0, 1.0, 1.0, 1.0]
