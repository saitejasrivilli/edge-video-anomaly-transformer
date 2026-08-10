"""Scaled dot-product attention and multi-head self-attention, from scratch.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Only ``nn.Linear``, ``nn.Dropout``, and raw tensor ops implement this —
there is no call to ``torch.nn.MultiheadAttention`` or
``torch.nn.TransformerEncoder`` anywhere in this module.

Masking semantics: a validity mask marks which **key** positions a query
is allowed to attend to. Invalid keys get their attention score set to
``-inf`` before softmax, so they receive exactly zero attention weight
regardless of query validity. Invalid **query** positions still produce
an output (attention over the valid keys), but that output is excluded
downstream by masked pooling (``pooling.py``) — this module does not need
to know or care which queries are "real," only which keys are.
"""

from __future__ import annotations

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention explicitly.

    Args:
        query: ``[B, H, Tq, D_head]``
        key: ``[B, H, Tk, D_head]``
        value: ``[B, H, Tk, D_head]``
        key_mask: ``[B, 1, 1, Tk]`` or ``[B, 1, Tq, Tk]`` bool — True where
            the key position is valid and may be attended to.
        dropout: applied to attention weights after softmax (standard
            Transformer dropout placement), only when in training mode.

    Returns:
        ``(output [B, H, Tq, D_head], attention_weights [B, H, Tq, Tk])``.
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)

    if key_mask is not None:
        scores = scores.masked_fill(~key_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    # A row can be entirely masked out (e.g. an all-padding sequence);
    # softmax over all -inf produces NaN, so replace with zero weight
    # rather than letting NaN propagate silently.
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention: project to Q/K/V, split heads, attend, merge, project out."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, T, D_head]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, T, H, D_head]
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``x``: ``[B, T, D]``. ``validity_mask``: ``[B, T]`` bool, True = valid key."""
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        key_mask = None
        if validity_mask is not None:
            key_mask = validity_mask[:, None, None, :]  # [B, 1, 1, T], broadcasts over H, Tq

        output, weights = scaled_dot_product_attention(
            q, k, v, key_mask=key_mask, dropout=self.dropout if self.training else None
        )
        merged = self._merge_heads(output)
        result = self.out_proj(merged)

        return result, (weights if return_attention else None)
