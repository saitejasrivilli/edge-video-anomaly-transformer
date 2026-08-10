"""Tracking data types.

``ObjectCandidate`` is deliberately anonymous — it carries a mask and an
optional bounding box for one frame, nothing else. It is what a
segmentation stage (Phase 3's U-Net, or, until that model is
instance-aware, per-instance regions extracted from a YouTube-VOS
object-ID mask) hands to the tracker. It never carries a ground-truth
identity: see ``evat.tracking.ground_truth`` for the separate, evaluation-
only representation that does.

``Track`` is the tracker's internal, mutable state for one predicted
identity across frames. ``TrackedInstance`` is the tracker's per-frame
output: which candidate got which predicted ``track_id`` this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

BoundingBox = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), inclusive-exclusive


def bbox_from_mask(mask: np.ndarray) -> BoundingBox | None:
    """Compute a tight bounding box from a binary mask, or None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


@dataclass(frozen=True, slots=True)
class ObjectCandidate:
    """One anonymous detected/segmented object instance in a single frame."""

    frame_id: str
    mask: np.ndarray  # [H, W] binary (0/1 or bool)
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        if self.bbox is None:
            object.__setattr__(self, "bbox", bbox_from_mask(self.mask))


class TrackState(Enum):
    """Lifecycle state of a predicted track.

    NEW -> created this frame from an unmatched candidate, not yet confirmed
        again.
    ACTIVE -> matched to a candidate on the most recent update (whether it
        was previously NEW, ACTIVE, or MISSED).
    MISSED -> not matched this frame, but within ``max_missed_frames`` of
        its last match; its last-known mask/bbox is retained so it can
        still be matched against in future frames.
    TERMINATED -> exceeded ``max_missed_frames`` consecutive misses; removed
        from the active track pool and will never be matched again.
    """

    NEW = "new"
    ACTIVE = "active"
    MISSED = "missed"
    TERMINATED = "terminated"


@dataclass(slots=True)
class Track:
    """Mutable per-identity tracker state."""

    track_id: int
    state: TrackState
    mask: np.ndarray
    bbox: BoundingBox | None
    last_frame_id: str
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    frame_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frame_history:
            self.frame_history.append(self.last_frame_id)


@dataclass(frozen=True, slots=True)
class TrackedInstance:
    """One tracker output: a candidate's assigned predicted track ID for one frame."""

    frame_id: str
    track_id: int
    state: TrackState
    mask: np.ndarray
    bbox: BoundingBox | None
