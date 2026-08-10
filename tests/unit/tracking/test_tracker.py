"""Synthetic tracker lifecycle tests. No YouTube-VOS data required."""

import numpy as np

from evat.tracking.schemas import ObjectCandidate, TrackState
from evat.tracking.tracker import Tracker, TrackerConfig


def _box_mask(x1, y1, x2, y2, shape=(20, 20)):
    m = np.zeros(shape, dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_case1_stable_object_gives_one_persistent_track():
    tracker = Tracker()
    mask = _box_mask(2, 2, 6, 6)

    ids = []
    for frame in range(5):
        outputs = tracker.update(str(frame), [ObjectCandidate(frame_id=str(frame), mask=mask)])
        ids.append(outputs[0].track_id)

    assert len(set(ids)) == 1


def test_case2_moving_object_keeps_same_identity():
    tracker = Tracker(TrackerConfig(iou_threshold=0.3))

    ids = []
    for frame, offset in enumerate([0, 1, 2, 3]):
        mask = _box_mask(2 + offset, 2, 8 + offset, 8)
        outputs = tracker.update(str(frame), [ObjectCandidate(frame_id=str(frame), mask=mask)])
        ids.append(outputs[0].track_id)

    assert len(set(ids)) == 1


def test_case3_two_distinct_objects_get_two_tracks():
    tracker = Tracker()
    mask_a = _box_mask(0, 0, 3, 3)
    mask_b = _box_mask(15, 15, 18, 18)

    ids_per_frame = []
    for frame in range(3):
        outputs = tracker.update(
            str(frame),
            [
                ObjectCandidate(frame_id=str(frame), mask=mask_a),
                ObjectCandidate(frame_id=str(frame), mask=mask_b),
            ],
        )
        ids_per_frame.append(sorted(o.track_id for o in outputs))

    assert all(ids == ids_per_frame[0] for ids in ids_per_frame)
    assert len(ids_per_frame[0]) == 2


def test_case4_temporary_disappearance_preserves_track_identity():
    tracker = Tracker(TrackerConfig(max_missed_frames=2))
    mask = _box_mask(2, 2, 6, 6)

    first = tracker.update("0", [ObjectCandidate(frame_id="0", mask=mask)])
    track_id = first[0].track_id

    # Object disappears for 2 frames (within max_missed_frames).
    tracker.update("1", [])
    tracker.update("2", [])

    live_track = next(t for t in tracker.tracks if t.track_id == track_id)
    assert live_track.state == TrackState.MISSED

    reappeared = tracker.update("3", [ObjectCandidate(frame_id="3", mask=mask)])
    assert reappeared[0].track_id == track_id
    assert reappeared[0].state == TrackState.ACTIVE


def test_disappearance_beyond_threshold_terminates_track():
    tracker = Tracker(TrackerConfig(max_missed_frames=1))
    mask = _box_mask(2, 2, 6, 6)

    first = tracker.update("0", [ObjectCandidate(frame_id="0", mask=mask)])
    track_id = first[0].track_id

    tracker.update("1", [])
    tracker.update("2", [])  # missed_frames now exceeds max_missed_frames=1

    live_track = next(t for t in tracker.tracks if t.track_id == track_id)
    assert live_track.state == TrackState.TERMINATED

    # A reappearing object at the same location now gets a NEW track id.
    reappeared = tracker.update("3", [ObjectCandidate(frame_id="3", mask=mask)])
    assert reappeared[0].track_id != track_id
    assert reappeared[0].state == TrackState.NEW


def test_case5_new_object_gets_new_track_id():
    tracker = Tracker()
    mask_a = _box_mask(0, 0, 3, 3)
    mask_b = _box_mask(15, 15, 18, 18)

    first = tracker.update("0", [ObjectCandidate(frame_id="0", mask=mask_a)])
    second = tracker.update(
        "1",
        [
            ObjectCandidate(frame_id="1", mask=mask_a),
            ObjectCandidate(frame_id="1", mask=mask_b),
        ],
    )

    new_object_output = next(o for o in second if o.track_id != first[0].track_id)
    assert new_object_output.state == TrackState.NEW


def test_case6_ambiguous_overlap_is_deterministic_across_runs():
    mask_a = _box_mask(0, 0, 5, 5)
    mask_b = _box_mask(2, 2, 7, 7)  # overlaps mask_a

    def run():
        tracker = Tracker()
        tracker.update("0", [ObjectCandidate(frame_id="0", mask=mask_a)])
        return tracker.update(
            "1",
            [
                ObjectCandidate(frame_id="1", mask=mask_a),
                ObjectCandidate(frame_id="1", mask=mask_b),
            ],
        )

    result_a = [(o.track_id, o.state) for o in run()]
    result_b = [(o.track_id, o.state) for o in run()]

    assert result_a == result_b


def test_case7_empty_frame_handled_cleanly():
    tracker = Tracker()

    outputs = tracker.update("0", [])

    assert outputs == []
    assert tracker.tracks == []


def test_finalize_filters_by_min_track_length():
    tracker = Tracker(TrackerConfig(min_track_length=2, max_missed_frames=0))
    mask = _box_mask(2, 2, 6, 6)

    tracker.update(
        "0", [ObjectCandidate(frame_id="0", mask=mask)]
    )  # 1 hit, then never matched again
    tracker.update("1", [])  # missed_frames=1 > max_missed_frames=0 -> terminated

    survivors = tracker.finalize()

    assert survivors == []
