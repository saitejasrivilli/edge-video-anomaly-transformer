"""Position-wise feed-forward network: Linear -> activation -> Dropout -> Linear."""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, activation: str = "relu", dropout: float = 0.0
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}, expected one of {list(_ACTIVATIONS)}"
            )

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
