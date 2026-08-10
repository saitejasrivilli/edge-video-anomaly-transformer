"""Visual feature encoders: a handcrafted baseline and a learned CNN backbone.

Both encoders implement the same contract: given ``[B, H, W, 3]`` uint8
crops (or a single ``[H, W, 3]`` crop), produce ``[B, D]`` (or ``[D]``)
float32 feature vectors. This shared contract is what lets Phase 6 (or
this phase's own comparison) swap encoders without touching the rest of
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# BaselineStatsEncoder: per-channel mean + std (6) + foreground area
# fraction (1) = 7 dims. A deliberately simple, fully deterministic,
# no-learning baseline that establishes the extraction interface before
# introducing a learned model.
BASELINE_FEATURE_DIM = 7


class BaselineStatsEncoder:
    """Handcrafted baseline: masked RGB mean/std + foreground area fraction."""

    name = "baseline_stats_v1"
    feature_dim = BASELINE_FEATURE_DIM

    def extract(self, crop_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Extract a 7-dim feature from one ``[H, W, 3]`` uint8 crop."""
        pixels = crop_rgb.reshape(-1, 3).astype(np.float64)

        if mask is not None:
            valid = mask.reshape(-1) != 0
            area_fraction = float(valid.mean()) if valid.size else 0.0
            pixels = pixels[valid] if valid.any() else pixels
        else:
            area_fraction = 1.0

        if pixels.size == 0:
            mean = np.zeros(3)
            std = np.zeros(3)
        else:
            mean = pixels.mean(axis=0)
            std = pixels.std(axis=0)

        feature = np.concatenate([mean, std, [area_fraction * 255.0]]) / 255.0
        return feature.astype(np.float32)

    def extract_batch(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.extract(c) for c in crops_rgb], axis=0)


@dataclass(slots=True)
class CNNEncoderConfig:
    """Configuration for ``CNNFeatureEncoder`` — nothing hard-coded.

    Backbone: MobileNetV3-Small (torchvision). Chosen over a larger
    ResNet for Colab/CPU feasibility (~2.5M parameters, ~2.9M for the
    feature-extractor portion used here) while still providing
    ImageNet-pretrained spatial features. Weights: torchvision's
    ``MobileNet_V3_Small_Weights.IMAGENET1K_V1``, license BSD-3-Clause
    (torchvision), downloaded automatically from PyTorch's model zoo on
    first use ONLY when ``pretrained=True`` — never during local tests
    or CI, which always construct with ``pretrained=False`` (random
    init, no network access) per CLAUDE.md Section 14/23.
    """

    backbone: str = "mobilenet_v3_small"
    pretrained: bool = False
    frozen: bool = True
    input_height: int = 128
    input_width: int = 128
    feature_dim: int | None = None  # None = use backbone's native output dim


class CNNFeatureEncoder(nn.Module):
    """Learned spatial feature encoder: CNN backbone (no classifier head) + pooling."""

    name = "mobilenet_v3_small"

    def __init__(self, config: CNNEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNEncoderConfig()

        if self.config.backbone != "mobilenet_v3_small":
            raise ValueError(f"Unsupported backbone: {self.config.backbone!r}")

        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # conv stack, no classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        native_dim = backbone.classifier[0].in_features  # 576 for mobilenet_v3_small

        self.projection = (
            nn.Linear(native_dim, self.config.feature_dim)
            if self.config.feature_dim is not None
            else nn.Identity()
        )
        self.feature_dim = self.config.feature_dim or native_dim

        if self.config.frozen:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``[B, 3, H, W]`` normalized float tensor -> ``[B, D]``."""
        with torch.set_grad_enabled(not self.config.frozen and self.training):
            x = self.features(images)
            x = self.pool(x).flatten(1)
        return self.projection(x)
