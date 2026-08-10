"""Object category classification task: label map, video-level split, dataset construction.

See docs/task_definition.md for the full task/leakage rationale. This
module only orchestrates existing, unmodified pipeline pieces:

    YouTube-VOS meta.json (Phase 2)
        -> ground-truth instances / tracker (Phase 4)
        -> object crops / visual features (Phase 5)
        -> TemporalFeatureSequence (Phase 5)
        -> CategorySample (this module: attaches the category label)
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from evat.data.datasets.youtube_vos import VideoRecord
from evat.features.encoders import BaselineStatsEncoder, CNNFeatureEncoder
from evat.features.extract import extract_object_features_baseline, extract_object_features_cnn
from evat.features.schemas import TemporalFeatureSequence
from evat.features.temporal import build_temporal_feature_sequence, group_features_by_track
from evat.tracking.ground_truth import (
    GroundTruthInstance,
    extract_ground_truth_instances,
    strip_identity,
)
from evat.tracking.matching import mask_iou
from evat.tracking.schemas import TrackedInstance
from evat.tracking.tracker import Tracker, TrackerConfig
from evat.video.sampling import uniform_frame_indices
from evat.video.sequence import build_temporal_sequence
from evat.video.tensors import load_temporal_sequence


def build_category_label_map(videos: list[VideoRecord]) -> dict[str, int]:
    """Map observed category names to integer class indices, deterministically.

    Sorted alphabetically rather than hard-coded, since the exact
    category set present depends on which videos are used.
    """
    categories = sorted({obj.category for video in videos for obj in video.objects})
    return {category: index for index, category in enumerate(categories)}


def split_videos_by_video_id(
    videos: list[VideoRecord], val_fraction: float, seed: int
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Split at video granularity so no video's frames appear in both splits.

    Raises:
        ValueError: if ``val_fraction`` is not in (0, 1).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    shuffled = list(videos)
    random.Random(seed).shuffle(shuffled)

    num_val = max(1, round(len(shuffled) * val_fraction))
    val_videos = shuffled[:num_val]
    train_videos = shuffled[num_val:]
    return train_videos, val_videos


@dataclass(frozen=True, slots=True)
class CategorySample:
    """One labeled training/evaluation example for the category-classification task."""

    video_id: str
    track_id: int
    category: str
    label: int
    sequence: TemporalFeatureSequence


def _majority_category(
    track_id: int,
    tracked_by_frame: dict[str, list[TrackedInstance]],
    gt_by_frame: dict[str, list[GroundTruthInstance]],
    iou_threshold: float = 0.5,
) -> str | None:
    """Recover a predicted track's ground-truth category via per-frame mask overlap.

    For each frame the track appears in, find the ground-truth instance
    whose mask best overlaps the track's predicted mask (mirroring the
    matching evat.tracking.metrics.evaluate_tracking uses for identity
    evaluation), then take the majority vote across frames. This is
    robust to multi-object frames, unlike assuming a fixed track-to-object
    ordering.
    """
    votes: Counter[str] = Counter()
    for frame_id, tracked_instances in tracked_by_frame.items():
        track_instance = next((t for t in tracked_instances if t.track_id == track_id), None)
        if track_instance is None:
            continue

        best_score, best_category = 0.0, None
        for gt_instance in gt_by_frame.get(frame_id, []):
            score = mask_iou(track_instance.mask, gt_instance.mask)
            if score >= iou_threshold and score > best_score:
                best_score, best_category = score, gt_instance.gt_object_id
        if best_category is not None:
            votes[best_category] += 1

    if not votes:
        return None
    return votes.most_common(1)[0][0]


def build_category_samples(
    videos: list[VideoRecord],
    dataset_root: str,
    label_map: dict[str, int],
    encoder: BaselineStatsEncoder | CNNFeatureEncoder,
    tracker_config: TrackerConfig | None = None,
    num_frames_per_video: int = 16,
    sequence_length: int | None = None,
    stride: int = 1,
) -> list[CategorySample]:
    """Build one ``CategorySample`` per tracked object across ``videos``.

    Reuses the Phase 4 tracker and Phase 5 crop/encode/temporal pipeline
    unchanged. Objects whose category is not in ``label_map`` are skipped
    (this happens when ``label_map`` was built from a different, e.g.
    train-only, video set than ``videos`` — a validation video should
    never introduce a new category never seen in training; see
    docs/task_definition.md).
    """
    samples: list[CategorySample] = []

    for video in videos:
        category_by_object_id = {obj.object_id: obj.category for obj in video.objects}

        num_samples = min(num_frames_per_video, len(video.frames))
        indices = uniform_frame_indices(num_frames_total=len(video.frames), num_samples=num_samples)
        sequence = build_temporal_sequence(video, indices)
        batch = load_temporal_sequence(sequence, dataset_root=dataset_root)

        tracker = Tracker(tracker_config)
        frame_order = list(batch.frame_ids)
        all_features = []
        tracked_by_frame: dict[str, list[TrackedInstance]] = {}
        gt_by_frame: dict[str, list[GroundTruthInstance]] = {}

        for i, frame_id in enumerate(frame_order):
            object_id_mask = batch.masks[i]
            gt_instances = (
                extract_ground_truth_instances(object_id_mask, frame_id)
                if object_id_mask is not None
                else []
            )
            gt_by_frame[frame_id] = gt_instances

            tracked = tracker.update(frame_id, strip_identity(gt_instances))
            tracked_by_frame[frame_id] = tracked
            frame_rgb = batch.images[i].transpose(1, 2, 0)

            if isinstance(encoder, CNNFeatureEncoder):
                all_features.extend(extract_object_features_cnn(frame_rgb, tracked, encoder))
            elif isinstance(encoder, BaselineStatsEncoder):
                all_features.extend(extract_object_features_baseline(frame_rgb, tracked, encoder))

        grouped = group_features_by_track(all_features)
        for track_id, features_by_frame in grouped.items():
            gt_object_id = _majority_category(track_id, tracked_by_frame, gt_by_frame)
            category = category_by_object_id.get(gt_object_id) if gt_object_id else None
            if category is None or category not in label_map:
                continue

            temporal_sequence = build_temporal_feature_sequence(
                track_id,
                frame_order,
                features_by_frame,
                sequence_length=sequence_length,
                stride=stride,
            )
            samples.append(
                CategorySample(
                    video_id=video.video_id,
                    track_id=track_id,
                    category=category,
                    label=label_map[category],
                    sequence=temporal_sequence,
                )
            )

    return samples
