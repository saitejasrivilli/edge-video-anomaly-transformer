"""Ground-truth object identity — kept structurally separate from tracker I/O.

YouTube-VOS annotation masks are palette-indexed: pixel value == object ID.
``extract_ground_truth_instances`` reads that ground truth directly.
``strip_identity`` is the only, explicit way to turn a ground-truth
instance into tracker input (``ObjectCandidate``) — it deliberately drops
the ID, so ground truth can never silently leak into what the tracker
"sees" as opposed to what evaluation later checks it against.

Note: until a segmentation model produces per-instance masks (Phase 3's
baseline is binary foreground/background only, not instance-aware), using
``strip_identity`` over YouTube-VOS ground-truth masks is how this phase
obtains per-frame object candidates to feed the tracker. This is
documented, not hidden: it stands in for a future instance-segmentation
stage, and evaluation never uses the stripped ID back — it re-derives
correctness purely from mask overlap between predicted tracks and
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evat.tracking.schemas import BoundingBox, ObjectCandidate, bbox_from_mask


@dataclass(frozen=True, slots=True)
class GroundTruthInstance:
    """One labeled ground-truth object instance in a single frame."""

    frame_id: str
    gt_object_id: str
    mask: np.ndarray  # [H, W] binary
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        if self.bbox is None:
            object.__setattr__(self, "bbox", bbox_from_mask(self.mask))


def extract_ground_truth_instances(
    object_id_mask: np.ndarray, frame_id: str
) -> list[GroundTruthInstance]:
    """Split a palette-indexed object-ID mask into one binary mask per object ID.

    Background (pixel value 0) is never treated as an object.
    """
    instances = []
    for object_id in sorted(int(v) for v in np.unique(object_id_mask) if v != 0):
        binary_mask = (object_id_mask == object_id).astype(np.uint8)
        instances.append(
            GroundTruthInstance(frame_id=frame_id, gt_object_id=str(object_id), mask=binary_mask)
        )
    return instances


def strip_identity(instances: list[GroundTruthInstance]) -> list[ObjectCandidate]:
    """Convert ground-truth instances into anonymous tracker input, dropping IDs."""
    return [ObjectCandidate(frame_id=i.frame_id, mask=i.mask, bbox=i.bbox) for i in instances]
