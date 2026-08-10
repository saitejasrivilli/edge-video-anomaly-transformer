"""Sinusoidal positional encoding, computed directly (not an external library call).

For position ``pos`` and dimension index ``2i`` / ``2i+1``:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Added elementwise to the input embeddings. Since attention itself is
permutation-invariant (it has no notion of order — swapping two input
tokens just swaps two rows of Q/K/V), this additive signal is what gives
the model any sense of temporal order at all.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [ceil(d_model/2)]

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, D]``. Adds positional encoding for positions ``0..T-1``.

        Raises:
            ValueError: if ``T`` exceeds ``max_len`` this encoding was built for.
        """
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[0]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional encoding max_len {self.pe.shape[0]}"
            )
        return x + self.pe[:seq_len].unsqueeze(0)
