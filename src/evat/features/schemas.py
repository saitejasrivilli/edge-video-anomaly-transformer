"""Feature representation types.

``VisualFeature`` is one feature vector for one frame — either object-level
(``track_id`` set, associated with a tracker identity from Phase 4) or
global/full-frame (``track_id is None``). ``TemporalFeatureSequence`` is
the ordered, per-track assembly of those vectors that a future Transformer
would consume, with an explicit validity mask so padded/missing positions
are never silently hidden from the downstream model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VisualFeature:
    """One extracted feature vector for one frame.

    Attributes:
        frame_id: which frame this feature was extracted from.
        track_id: the Phase 4 tracker identity this feature represents, or
            None for a global/full-frame feature (not tied to any object).
        feature: ``[D]`` float32 vector.
        extractor_name: identifies which extractor produced this vector
            (e.g. "baseline_stats_v1", "mobilenet_v3_small_frozen") —
            needed because features from different extractors/configs are
            not comparable and must not be silently mixed.
    """

    frame_id: str
    track_id: int | None
    feature: np.ndarray
    extractor_name: str

    @property
    def feature_dim(self) -> int:
        return int(self.feature.shape[-1])


@dataclass(frozen=True, slots=True)
class TemporalFeatureSequence:
    """An ordered, fixed-length feature sequence for one track.

    ``features[t]`` is only meaningful where ``validity[t]`` is True.
    Invalid positions hold a zero vector — a real value is never
    fabricated for a missing frame (see docs/architecture.md, "Missing
    frames").

    Attributes:
        track_id: which tracked object this sequence belongs to.
        frame_ids: the frame ID at each position (``None`` where padded).
        features: ``[T, D]`` float32.
        validity: ``[T]`` bool — True where ``features[t]`` is a real,
            extracted feature; False where it is padding.
    """

    track_id: int
    frame_ids: tuple[str | None, ...]
    features: np.ndarray
    validity: np.ndarray

    @property
    def length(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])
