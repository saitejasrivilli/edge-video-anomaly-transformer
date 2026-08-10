import numpy as np

from evat.tracking.matching import bbox_iou, compute_score_matrix, greedy_match, mask_iou
from evat.tracking.schemas import ObjectCandidate, Track, TrackState


def _mask(coords, shape=(4, 4)):
    m = np.zeros(shape, dtype=np.uint8)
    for y, x in coords:
        m[y, x] = 1
    return m


def test_mask_iou_identical_masks_is_one():
    m = _mask([(0, 0), (0, 1)])
    assert mask_iou(m, m) == 1.0


def test_mask_iou_disjoint_masks_is_zero():
    a = _mask([(0, 0)])
    b = _mask([(3, 3)])
    assert mask_iou(a, b) == 0.0


def test_bbox_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert bbox_iou(box, box) == 1.0


def test_bbox_iou_none_is_zero():
    assert bbox_iou(None, (0, 0, 5, 5)) == 0.0


def test_greedy_match_is_deterministic_and_respects_threshold():
    tracks = [
        Track(
            track_id=1,
            state=TrackState.ACTIVE,
            mask=_mask([(0, 0)]),
            bbox=(0, 0, 1, 1),
            last_frame_id="0",
        ),
    ]
    candidates = [
        ObjectCandidate(frame_id="1", mask=_mask([(0, 0)])),
        ObjectCandidate(frame_id="1", mask=_mask([(3, 3)])),
    ]

    scores = compute_score_matrix(tracks, candidates, "mask_iou")
    matches = greedy_match(scores, threshold=0.5)

    assert matches == [(0, 0)]


def test_greedy_match_breaks_ties_deterministically():
    scores = np.array([[0.5, 0.5]])

    matches = greedy_match(scores, threshold=0.3)

    # Tie broken by lowest candidate index.
    assert matches == [(0, 0)]
