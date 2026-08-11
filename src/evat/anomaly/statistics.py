"""Normal feature distribution fitting and Mahalanobis distance.

Method: fit a single multivariate Gaussian (mean + covariance) over
normal-only training features, then score any feature vector by its
Mahalanobis distance from that distribution — the standard "distance from
normality" anomaly score. Chosen over a reconstruction-based method
(e.g. autoencoder) for implementation simplicity, interpretability (a
distance is directly explainable), and because it works directly on top
of the already-pretrained Phase 5 CNN encoder without any additional
training.

Regularization: with MVTec's few-hundred-image training sets and a
576-dim (MobileNetV3-Small native) feature space, the sample covariance
is frequently near-singular. A small ``eps * I`` is added to the diagonal
before inverting (shrinkage regularization) — a standard, documented fix,
not a hidden numerical hack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalDistribution:
    """A fitted multivariate Gaussian over normal-only features."""

    mean: np.ndarray  # [D]
    precision: np.ndarray  # [D, D], regularized inverse covariance

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])


def fit_normal_distribution(features: np.ndarray, eps: float = 1e-3) -> NormalDistribution:
    """Fit mean + regularized precision matrix from ``[N, D]`` normal-only features.

    Raises:
        ValueError: if fewer than 2 samples are given (covariance undefined).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}")
    if features.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to fit a covariance, got {features.shape[0]}")

    mean = features.mean(axis=0)
    centered = features - mean
    covariance = (centered.T @ centered) / (features.shape[0] - 1)
    regularized = covariance + eps * np.eye(covariance.shape[0])
    precision = np.linalg.inv(regularized)

    return NormalDistribution(mean=mean.astype(np.float64), precision=precision)


def mahalanobis_distance(features: np.ndarray, distribution: NormalDistribution) -> np.ndarray:
    """Mahalanobis distance of each row of ``[N, D]`` (or a single ``[D]``) from ``distribution``.

    distance(x) = sqrt((x - mean)^T @ precision @ (x - mean))
    """
    single = features.ndim == 1
    x = features[None, :] if single else features

    centered = x - distribution.mean
    distances = np.sqrt(np.einsum("ni,ij,nj->n", centered, distribution.precision, centered))
    distances = np.clip(distances, 0, None)  # guard tiny negative values from float error

    return distances[0] if single else distances
