"""Temporal aggregation: masked mean pooling over the sequence dimension.

    pooled = sum_t(valid_t * x_t) / count(valid_t)

Padded/invalid positions never contribute to the sum or the denominator —
this is why the Phase 5 validity mask must be threaded all the way
through the encoder to this point, not just used for attention masking.
"""

from __future__ import annotations

import torch


def masked_mean_pool(x: torch.Tensor, validity_mask: torch.Tensor | None = None) -> torch.Tensor:
    """``x``: ``[B, T, D]``. ``validity_mask``: ``[B, T]`` bool, True = valid.

    If ``validity_mask`` is None, every position is treated as valid
    (plain mean pooling). Sequences with zero valid positions produce a
    zero vector rather than dividing by zero.
    """
    if validity_mask is None:
        return x.mean(dim=1)

    mask = validity_mask.unsqueeze(-1).float()  # [B, T, 1]
    summed = (x * mask).sum(dim=1)  # [B, D]
    counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return summed / counts
