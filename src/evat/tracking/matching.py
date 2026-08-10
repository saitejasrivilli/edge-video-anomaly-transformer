"""Track-to-candidate matching: scoring and deterministic greedy assignment.

Matching logic is implemented directly here (mask IoU / bbox IoU / centroid
distance) rather than delegated to an opaque external tracking library, so
the matching behavior is fully inspectable and testable.
"""

from __future__ import annotations

import numpy as np

from evat.tracking.schemas import BoundingBox, ObjectCandidate, Track

MatchingMethod = str  # "mask_iou" | "bbox_iou" | "centroid"


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union between two binary masks of the same shape."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def bbox_iou(bbox_a: BoundingBox | None, bbox_b: BoundingBox | None) -> float:
    """Intersection-over-union between two ``(x_min, y_min, x_max, y_max)`` boxes."""
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return intersection / union


def centroid_similarity(bbox_a: BoundingBox | None, bbox_b: BoundingBox | None) -> float:
    """A [0, 1] similarity score, higher for closer box centroids.

    Not a raw distance (which would be unbounded and hard to threshold
    consistently against IoU-based scores): converted to
    ``1 / (1 + distance)`` so it composes with the same
    ``score >= threshold`` matching logic as the IoU-based methods.
    """
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax = (bbox_a[0] + bbox_a[2]) / 2
    ay = (bbox_a[1] + bbox_a[3]) / 2
    bx = (bbox_b[0] + bbox_b[2]) / 2
    by = (bbox_b[1] + bbox_b[3]) / 2
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return 1.0 / (1.0 + distance)


def compute_score_matrix(
    tracks: list[Track], candidates: list[ObjectCandidate], method: MatchingMethod
) -> np.ndarray:
    """``[num_tracks, num_candidates]`` matrix of matching scores in [0, 1]."""
    scores = np.zeros((len(tracks), len(candidates)), dtype=np.float64)
    for i, track in enumerate(tracks):
        for j, candidate in enumerate(candidates):
            if method == "mask_iou":
                scores[i, j] = mask_iou(track.mask, candidate.mask)
            elif method == "bbox_iou":
                scores[i, j] = bbox_iou(track.bbox, candidate.bbox)
            elif method == "centroid":
                scores[i, j] = centroid_similarity(track.bbox, candidate.bbox)
            else:
                raise ValueError(f"Unsupported matching method: {method!r}")
    return scores


def greedy_match(score_matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Deterministic greedy assignment: highest score first, ties broken by index.

    Returns a list of ``(track_index, candidate_index)`` pairs. Each track
    and each candidate appears in at most one pair. This is not globally
    optimal (unlike the Hungarian algorithm) but is simple, fast, and
    fully deterministic — adequate for a first tracking baseline.
    """
    num_tracks, num_candidates = score_matrix.shape
    candidates_pairs = [
        (score_matrix[i, j], i, j) for i in range(num_tracks) for j in range(num_candidates)
    ]
    candidates_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_tracks: set[int] = set()
    matched_candidates: set[int] = set()
    matches: list[tuple[int, int]] = []

    for score, track_idx, candidate_idx in candidates_pairs:
        if score < threshold:
            break
        if track_idx in matched_tracks or candidate_idx in matched_candidates:
            continue
        matches.append((track_idx, candidate_idx))
        matched_tracks.add(track_idx)
        matched_candidates.add(candidate_idx)

    return matches
