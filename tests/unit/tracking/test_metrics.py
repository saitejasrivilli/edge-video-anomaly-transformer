import numpy as np

from evat.tracking.ground_truth import GroundTruthInstance
from evat.tracking.metrics import evaluate_tracking
from evat.tracking.schemas import TrackedInstance, TrackState


def _mask(x1, y1, x2, y2, shape=(10, 10)):
    m = np.zeros(shape, dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_perfect_tracking_gives_full_coverage_and_consistency():
    mask = _mask(0, 0, 3, 3)
    frames = ["0", "1", "2"]
    gt = {f: [GroundTruthInstance(frame_id=f, gt_object_id="1", mask=mask)] for f in frames}
    predictions = {
        f: [TrackedInstance(frame_id=f, track_id=7, state=TrackState.ACTIVE, mask=mask, bbox=None)]
        for f in frames
    }

    metrics = evaluate_tracking(gt, predictions, frame_order=frames)

    assert metrics.coverage == 1.0
    assert metrics.id_consistency == 1.0
    assert metrics.identity_switches == 0
    assert metrics.track_fragmentation == 1.0


def test_identity_switch_is_detected():
    mask = _mask(0, 0, 3, 3)
    frames = ["0", "1"]
    gt = {f: [GroundTruthInstance(frame_id=f, gt_object_id="1", mask=mask)] for f in frames}
    predictions = {
        "0": [
            TrackedInstance(frame_id="0", track_id=1, state=TrackState.ACTIVE, mask=mask, bbox=None)
        ],
        "1": [
            TrackedInstance(frame_id="1", track_id=2, state=TrackState.NEW, mask=mask, bbox=None)
        ],
    }

    metrics = evaluate_tracking(gt, predictions, frame_order=frames)

    assert metrics.identity_switches == 1
    assert metrics.track_fragmentation == 2.0
    assert metrics.id_consistency == 0.5


def test_no_predictions_gives_zero_coverage():
    mask = _mask(0, 0, 3, 3)
    frames = ["0"]
    gt = {f: [GroundTruthInstance(frame_id=f, gt_object_id="1", mask=mask)] for f in frames}
    predictions: dict = {"0": []}

    metrics = evaluate_tracking(gt, predictions, frame_order=frames)

    assert metrics.coverage == 0.0
    assert metrics.identity_switches == 0
    assert metrics.track_fragmentation == 0.0
