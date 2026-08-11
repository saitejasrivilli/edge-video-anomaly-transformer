"""Per-category anomaly model: fitting, image-level scoring, pixel-level anomaly maps.

Category strategy: ONE model PER CATEGORY, never a single global model
across categories. MVTec categories (e.g. "bottle" vs. "screw") have
completely different normal appearances — a global feature distribution
would treat ordinary cross-category variation as "anomalous," which is
not a valid anomaly-detection setup. This mirrors the dataset's own
per-category train/test structure and the standard MVTec AD evaluation
protocol. See docs/anomaly_task_definition.md.

Pixel-level localization is intentionally coarse: ONE shared normal
distribution is fit over feature vectors from ALL spatial positions of
the backbone's feature map (not a separate distribution per spatial
position, as in more elaborate patch-based methods like PaDiM). This is
a documented simplification for implementation simplicity and small
per-category training-set sizes — the resulting anomaly map has the
resolution of the backbone's downsampled feature grid (coarse regions,
not pixel-precise segmentation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evat.anomaly.spatial_features import extract_pooled_features, extract_spatial_features
from evat.anomaly.statistics import (
    NormalDistribution,
    fit_normal_distribution,
    mahalanobis_distance,
)
from evat.features.encoders import CNNFeatureEncoder


@dataclass(frozen=True, slots=True)
class CategoryAnomalyModel:
    """Fitted normality model for exactly one MVTec category."""

    category: str
    image_level: NormalDistribution
    pixel_level: NormalDistribution


def fit_category_anomaly_model(
    category: str, encoder: CNNFeatureEncoder, normal_images: torch.Tensor, eps: float = 1e-3
) -> CategoryAnomalyModel:
    """Fit a category's normality model from its normal (``good``) TRAINING images only.

    Args:
        normal_images: ``[N, 3, H, W]`` normalized normal-only training images.

    Raises:
        ValueError: if fewer than 2 images are given.
    """
    pooled = extract_pooled_features(encoder, normal_images).cpu().numpy()  # [N, D]
    image_level = fit_normal_distribution(pooled, eps=eps)

    spatial = extract_spatial_features(encoder, normal_images).cpu().numpy()  # [N, C, H', W']
    n, c, h, w = spatial.shape
    spatial_vectors = spatial.transpose(0, 2, 3, 1).reshape(n * h * w, c)  # [N*H'*W', C]
    pixel_level = fit_normal_distribution(spatial_vectors, eps=eps)

    return CategoryAnomalyModel(category=category, image_level=image_level, pixel_level=pixel_level)


def image_anomaly_score(
    model: CategoryAnomalyModel, encoder: CNNFeatureEncoder, images: torch.Tensor
) -> np.ndarray:
    """``images``: ``[N, 3, H, W]`` -> ``[N]`` anomaly scores (higher = more anomalous)."""
    pooled = extract_pooled_features(encoder, images).cpu().numpy()
    return mahalanobis_distance(pooled, model.image_level)


def anomaly_map(
    model: CategoryAnomalyModel, encoder: CNNFeatureEncoder, image: torch.Tensor
) -> np.ndarray:
    """``image``: ``[1, 3, H, W]`` -> ``[H', W']`` anomaly map at feature-grid resolution."""
    spatial = extract_spatial_features(encoder, image).cpu().numpy()  # [1, C, H', W']
    _, c, h, w = spatial.shape
    vectors = spatial[0].transpose(1, 2, 0).reshape(h * w, c)  # [H'*W', C]
    distances = mahalanobis_distance(vectors, model.pixel_level)
    return distances.reshape(h, w)
