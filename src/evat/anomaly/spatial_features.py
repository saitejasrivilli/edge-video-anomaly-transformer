"""Spatial (pre-pool) feature maps from the Phase 5 CNN backbone, for pixel-level localization.

``CNNFeatureEncoder.forward`` (Phase 5, unmodified) returns a single
pooled ``[B, D]`` vector per image — appropriate for image-level anomaly
scoring, but it discards spatial layout entirely, so it cannot support
localization. This module reuses the same encoder's already-public
``.features`` conv stack (nothing in Phase 5 is modified) to get the
pre-pool spatial map instead.
"""

from __future__ import annotations

import torch

from evat.features.encoders import CNNFeatureEncoder


@torch.no_grad()
def extract_spatial_features(encoder: CNNFeatureEncoder, images: torch.Tensor) -> torch.Tensor:
    """``images``: ``[B, 3, H, W]`` normalized -> ``[B, C, H', W']`` spatial feature map."""
    encoder.eval()
    return encoder.features(images)


@torch.no_grad()
def extract_pooled_features(encoder: CNNFeatureEncoder, images: torch.Tensor) -> torch.Tensor:
    """``images``: ``[B, 3, H, W]`` -> ``[B, D]`` pooled feature vector."""
    encoder.eval()
    return encoder(images)
