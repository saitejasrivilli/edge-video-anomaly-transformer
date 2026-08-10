import numpy as np

from evat.tracking.schemas import ObjectCandidate
from evat.tracking.tracker import Tracker, TrackerConfig


def _box_mask(x1, y1, x2, y2, shape=(10, 10)):
    m = np.zeros(shape, dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def _run_sequence(config: TrackerConfig) -> list[list[tuple[int, str]]]:
    tracker = Tracker(config)
    sequence = [
        [ObjectCandidate(frame_id="0", mask=_box_mask(0, 0, 3, 3))],
        [
            ObjectCandidate(frame_id="1", mask=_box_mask(1, 0, 4, 3)),
            ObjectCandidate(frame_id="1", mask=_box_mask(6, 6, 9, 9)),
        ],
        [],
        [ObjectCandidate(frame_id="3", mask=_box_mask(2, 0, 5, 3))],
    ]
    results = []
    for frame_id, candidates in enumerate(sequence):
        outputs = tracker.update(str(frame_id), candidates)
        results.append([(o.track_id, o.state.value) for o in outputs])
    return results


def test_same_input_and_config_produces_same_track_assignments():
    config = TrackerConfig(iou_threshold=0.2, max_missed_frames=2)

    run1 = _run_sequence(config)
    run2 = _run_sequence(config)

    assert run1 == run2
