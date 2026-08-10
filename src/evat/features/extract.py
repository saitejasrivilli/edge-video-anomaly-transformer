"""Ties crops + encoders to Phase 4 tracked instances, producing VisualFeatures.

Object-level features are extracted per ``TrackedInstance`` (so they carry
the Phase 4 ``track_id``); a separate function extracts one global
full-frame feature (``track_id=None``) when a scene-level representation
is also wanted (see docs/architecture.md for when to use each).
"""

from __future__ import annotations

import numpy as np

from evat.features.crops import crop_object, resize_crop
from evat.features.encoders import BaselineStatsEncoder, CNNFeatureEncoder
from evat.features.schemas import VisualFeature
from evat.tracking.schemas import TrackedInstance


def extract_object_features_baseline(
    frame_rgb: np.ndarray,
    instances: list[TrackedInstance],
    encoder: BaselineStatsEncoder,
    padding: int = 0,
    mask_aware: bool = True,
) -> list[VisualFeature]:
    """Extract one baseline feature per tracked instance in one frame.

    Instances with no bbox (empty mask) are skipped, not fabricated.
    """
    features = []
    for instance in instances:
        if instance.bbox is None:
            continue
        crop = crop_object(
            frame_rgb, instance.bbox, padding=padding, mask=instance.mask if mask_aware else None
        )
        feature_vector = encoder.extract(crop, mask=None)
        features.append(
            VisualFeature(
                frame_id=instance.frame_id,
                track_id=instance.track_id,
                feature=feature_vector,
                extractor_name=encoder.name,
            )
        )
    return features


def extract_object_features_cnn(
    frame_rgb: np.ndarray,
    instances: list[TrackedInstance],
    encoder: CNNFeatureEncoder,
    padding: int = 0,
    mask_aware: bool = True,
) -> list[VisualFeature]:
    """Extract one learned-CNN feature per tracked instance in one frame.

    Crops are resized to ``encoder.config.input_height/width`` and
    normalized to ``[0, 1]`` (this is a spatial visual encoder, not the
    segmentation pipeline — see ``evat.features.crops`` for why this
    preprocessing is kept separate from ``evat.training.transforms``).
    """
    import torch

    valid_instances = [i for i in instances if i.bbox is not None]
    if not valid_instances:
        return []

    size = (encoder.config.input_height, encoder.config.input_width)
    batch = []
    for instance in valid_instances:
        assert instance.bbox is not None  # guaranteed by valid_instances filter
        crop = crop_object(
            frame_rgb, instance.bbox, padding=padding, mask=instance.mask if mask_aware else None
        )
        resized = resize_crop(crop, size)
        batch.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)

    images = torch.stack(batch, dim=0)
    encoder.eval()
    with torch.no_grad():
        vectors = encoder(images).cpu().numpy()

    return [
        VisualFeature(
            frame_id=instance.frame_id,
            track_id=instance.track_id,
            feature=vectors[i],
            extractor_name=encoder.name,
        )
        for i, instance in enumerate(valid_instances)
    ]


def extract_global_feature_baseline(
    frame_rgb: np.ndarray, frame_id: str, encoder: BaselineStatsEncoder
) -> VisualFeature:
    """One full-frame (not object-specific) baseline feature, track_id=None."""
    return VisualFeature(
        frame_id=frame_id,
        track_id=None,
        feature=encoder.extract(frame_rgb, mask=None),
        extractor_name=encoder.name,
    )
