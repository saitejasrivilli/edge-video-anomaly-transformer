"""Simple recurrent temporal baseline: GRU over the feature sequence -> masked mean pool -> MLP.

Exists to answer, in Phase 7: does the Transformer provide value beyond a
standard (non-attention) temporal model, not just over a non-temporal
mean-pool baseline (``evat.models.temporal_baseline``)?

Pooling choice: rather than taking the GRU's final hidden state (which
assumes valid frames are a contiguous prefix — not guaranteed when a
tracked object has occlusion gaps in the middle of a sequence, per Phase
5's missing-frame handling), this baseline runs the GRU over the full
padded sequence and applies the same ``masked_mean_pool`` the Transformer
uses. This keeps the comparison controlled (both models pool the same
way) and correctly ignores padded positions regardless of where they
fall in the sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from evat.models.transformer.pooling import masked_mean_pool


class TemporalGRUBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, features: torch.Tensor, validity_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``features``: ``[B, T, feature_dim]`` -> ``[B, num_classes]``."""
        hidden_states, _ = self.gru(features)  # [B, T, hidden_dim]
        pooled = masked_mean_pool(hidden_states, validity_mask)
        return self.head(pooled)
