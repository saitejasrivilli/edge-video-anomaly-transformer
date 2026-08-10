"""Tracking evaluation: ground truth vs. predicted track identities.

Deliberately does NOT claim MOTA/HOTA/IDF1 — this baseline implements
simpler, clearly-defined identity metrics instead:

- **coverage**: fraction of ground-truth object-frames for which some
  predicted track overlaps it (mask IoU >= threshold).
- **id_consistency**: for each ground-truth object, the fraction of its
  matched frames assigned to that object's single most common predicted
  track ID (1.0 = perfectly consistent identity; lower = fragmented across
  multiple predicted IDs), averaged across ground-truth objects.
- **identity_switches**: total number of frame-to-frame changes in the
  matched predicted track ID, summed across all ground-truth objects.
- **track_fragmentation**: average number of *distinct* predicted track
  IDs matched to each ground-truth object (1.0 = perfect, no
  fragmentation).

Ground truth (``GroundTruthInstance.gt_object_id``) and predictions
(``TrackedInstance.track_id``) are read from two independently-typed
inputs — the ground-truth ID is never treated as a tracker output.
"""

from __future__ import annotations

from dataclasses import dataclass

from evat.tracking.ground_truth import GroundTruthInstance
from evat.tracking.matching import mask_iou
from evat.tracking.schemas import TrackedInstance


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    coverage: float
    id_consistency: float
    identity_switches: int
    track_fragmentation: float


def _best_match(
    gt_instance: GroundTruthInstance, predictions: list[TrackedInstance], iou_threshold: float
) -> int | None:
    best_score = 0.0
    best_track_id: int | None = None
    for pred in predictions:
        score = mask_iou(gt_instance.mask, pred.mask)
        if score >= iou_threshold and score > best_score:
            best_score = score
            best_track_id = pred.track_id
    return best_track_id


def evaluate_tracking(
    gt_by_frame: dict[str, list[GroundTruthInstance]],
    predictions_by_frame: dict[str, list[TrackedInstance]],
    frame_order: list[str],
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """Compare predicted tracks against ground truth over an ordered sequence of frames."""
    matched_ids_per_gt_object: dict[str, list[int]] = {}
    total_gt_instances = 0
    total_matched = 0

    for frame_id in frame_order:
        gt_instances = gt_by_frame.get(frame_id, [])
        predictions = predictions_by_frame.get(frame_id, [])
        for gt_instance in gt_instances:
            total_gt_instances += 1
            matched_track_id = _best_match(gt_instance, predictions, iou_threshold)
            if matched_track_id is not None:
                total_matched += 1
                matched_ids_per_gt_object.setdefault(gt_instance.gt_object_id, []).append(
                    matched_track_id
                )

    coverage = total_matched / total_gt_instances if total_gt_instances else 0.0

    identity_switches = 0
    consistency_scores = []
    fragmentation_counts = []

    for track_id_sequence in matched_ids_per_gt_object.values():
        for prev_id, curr_id in zip(track_id_sequence, track_id_sequence[1:], strict=False):
            if prev_id != curr_id:
                identity_switches += 1

        counts: dict[int, int] = {}
        for tid in track_id_sequence:
            counts[tid] = counts.get(tid, 0) + 1
        dominant_count = max(counts.values())
        consistency_scores.append(dominant_count / len(track_id_sequence))
        fragmentation_counts.append(len(counts))

    id_consistency = (
        sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    )
    track_fragmentation = (
        sum(fragmentation_counts) / len(fragmentation_counts) if fragmentation_counts else 0.0
    )

    return TrackingMetrics(
        coverage=coverage,
        id_consistency=id_consistency,
        identity_switches=identity_switches,
        track_fragmentation=track_fragmentation,
    )
