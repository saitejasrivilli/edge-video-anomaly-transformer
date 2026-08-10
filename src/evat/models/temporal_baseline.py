"""Simple non-temporal-attention baseline: masked mean-pool visual features -> MLP -> logits.

Exists to answer, in Phase 7: does the Transformer's self-attention add
useful modeling capacity over just averaging the per-frame features? The
comparison EXPERIMENT itself runs in Phase 7 — this phase only implements
the baseline interface so that experiment is easy to run later.
"""

from __future__ import annotations

import torch
from torch import nn

from evat.models.transformer.pooling import masked_mean_pool


class TemporalMeanPoolBaseline(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, features: torch.Tensor, validity_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``features``: ``[B, T, feature_dim]`` -> ``[B, num_classes]``."""
        pooled = masked_mean_pool(features, validity_mask)
        return self.mlp(pooled)
