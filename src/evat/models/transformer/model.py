"""Full VideoTransformer model: encoder -> masked mean pool -> classification head.

The prediction head here does classification, but the interface
(``VideoTransformerOutput`` exposing pooled + per-timestep representations,
not just logits) is deliberately reusable for later heads — anomaly
scoring, temporal event prediction — without changing the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from evat.models.transformer.config import TransformerConfig
from evat.models.transformer.encoder import TransformerEncoder
from evat.models.transformer.pooling import masked_mean_pool


@dataclass
class VideoTransformerOutput:
    """Named outputs — avoids an anonymous tuple with positional meaning.

    Attributes:
        logits: ``[B, num_classes]``.
        pooled: ``[B, d_model]`` — the masked-mean-pooled representation
            the classification head was computed from.
        temporal_representations: ``[B, T, d_model]`` — per-timestep
            encoder output, exposed for later experimentation (e.g. a
            future anomaly-detection head operating per-frame instead of
            per-sequence).
        attention_weights: list of ``[B, H, T, T]`` per layer, or None
            when not requested (``return_attention=False``) — not
            returned by default to avoid the extra memory cost.
    """

    logits: torch.Tensor
    pooled: torch.Tensor
    temporal_representations: torch.Tensor
    attention_weights: list[torch.Tensor] | None


class VideoTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        features: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> VideoTransformerOutput:
        """``features``: ``[B, T, feature_dim]``. ``validity_mask``: ``[B, T]`` bool."""
        temporal_representations, attention_weights = self.encoder(
            features, validity_mask=validity_mask, return_attention=return_attention
        )
        pooled = masked_mean_pool(temporal_representations, validity_mask)
        logits = self.classifier(pooled)

        return VideoTransformerOutput(
            logits=logits,
            pooled=pooled,
            temporal_representations=temporal_representations,
            attention_weights=attention_weights,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
