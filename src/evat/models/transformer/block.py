"""Transformer encoder block: pre-norm residual attention + pre-norm residual FFN.

    x = x + MHSA(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Pre-norm (normalize before the sublayer, not after) is used deliberately:
post-norm (the original "Attention Is All You Need" convention) is known
to need careful learning-rate warmup to train stably as depth increases,
because gradients flow through the un-normalized residual stream less
predictably. Pre-norm keeps the residual path identity-like end to end,
which trains more reliably without a warmup schedule — a better default
for a small, from-scratch baseline. This choice is applied consistently
in every block; the two conventions are not mixed.
"""

from __future__ import annotations

import torch
from torch import nn

from evat.models.transformer.attention import MultiHeadSelfAttention
from evat.models.transformer.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff, activation=activation, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attention(
            self.norm1(x), validity_mask=validity_mask, return_attention=return_attention
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x, attn_weights
