"""Stack of Transformer blocks: input projection -> positional encoding -> blocks -> final norm."""

from __future__ import annotations

import torch
from torch import nn

from evat.models.transformer.block import TransformerBlock
from evat.models.transformer.config import TransformerConfig
from evat.models.transformer.positional import SinusoidalPositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = (
            nn.Linear(config.feature_dim, config.d_model)
            if config.feature_dim != config.d_model
            else nn.Identity()
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, max_len=config.max_sequence_length
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        features: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """``features``: ``[B, T, feature_dim]`` -> ``[B, T, d_model]``."""
        x = self.input_projection(features)
        x = self.positional_encoding(x)

        attention_maps: list[torch.Tensor] | None = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(
                x, validity_mask=validity_mask, return_attention=return_attention
            )
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)  # type: ignore[union-attr]

        return self.final_norm(x), attention_maps
