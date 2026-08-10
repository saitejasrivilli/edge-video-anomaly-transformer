"""Temporal feature-sequence builder.

Converts per-frame, per-track ``VisualFeature``s into ordered
``TemporalFeatureSequence``s ([T, D] + a [T] validity mask), for one track
at a time.

Missing-frame strategy (documented choice, see docs/architecture.md):
when a track has no feature for a given frame position (e.g. Phase 4
marked it MISSED, or it simply wasn't matched that frame), that position
is **padded with a zero vector and marked invalid** in the validity mask
— never interpolated or fabricated. This keeps the sequence a fixed,
frame-aligned length while making padding explicit to any downstream
model (a future Transformer would use the validity mask as an attention
mask).
"""

from __future__ import annotations

import numpy as np

from evat.features.schemas import TemporalFeatureSequence, VisualFeature


def group_features_by_track(
    features: list[VisualFeature],
) -> dict[int, dict[str, VisualFeature]]:
    """Group object-level features (track_id is not None) by track, then frame_id."""
    grouped: dict[int, dict[str, VisualFeature]] = {}
    for feature in features:
        if feature.track_id is None:
            continue
        grouped.setdefault(feature.track_id, {})[feature.frame_id] = feature
    return grouped


def build_temporal_feature_sequence(
    track_id: int,
    frame_order: list[str],
    features_by_frame: dict[str, VisualFeature],
    sequence_length: int | None = None,
    stride: int = 1,
) -> TemporalFeatureSequence:
    """Build one track's temporal feature sequence over ``frame_order``.

    Args:
        track_id: which track this sequence is for.
        frame_order: the full, sorted list of frame IDs in the video/window
            being processed (from the Phase 2 temporal sequence) — this
            defines valid vs. missing positions, not just which features
            happen to exist.
        features_by_frame: this track's available features, keyed by frame_id
            (typically ``group_features_by_track(...)[track_id]``).
        sequence_length: if given, the output has exactly this many
            positions — truncated if longer, zero-padded (marked invalid)
            at the end if shorter. If None, uses ``len(frame_order[::stride])``.
        stride: sample every ``stride``-th frame from ``frame_order``
            (consistent with ``evat.video.sampling.strided_frame_indices``).

    Raises:
        ValueError: if no feature is available anywhere (feature dimension
            cannot be inferred) or if ``stride`` < 1.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if not features_by_frame:
        raise ValueError(
            f"No features available for track {track_id}; cannot infer feature dimension"
        )

    feature_dim = next(iter(features_by_frame.values())).feature_dim
    sampled_frames = frame_order[::stride]

    if sequence_length is None:
        sequence_length = len(sampled_frames)

    frame_ids: list[str | None] = list(sampled_frames[:sequence_length])
    while len(frame_ids) < sequence_length:
        frame_ids.append(None)

    features = np.zeros((sequence_length, feature_dim), dtype=np.float32)
    validity = np.zeros(sequence_length, dtype=bool)

    for i, frame_id in enumerate(frame_ids):
        if frame_id is not None and frame_id in features_by_frame:
            features[i] = features_by_frame[frame_id].feature
            validity[i] = True

    return TemporalFeatureSequence(
        track_id=track_id,
        frame_ids=tuple(frame_ids),
        features=features,
        validity=validity,
    )
