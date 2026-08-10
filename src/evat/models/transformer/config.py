"""Transformer configuration — every architecture parameter is explicit, nothing hard-coded."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class TransformerConfig:
    """All knobs needed to construct and reproduce a ``VideoTransformer``.

    Raises (in ``__post_init__``):
        ValueError: for any invalid combination (see individual checks).
    """

    feature_dim: int = 576  # matches CNNFeatureEncoder's native MobileNetV3-Small output
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 64
    num_classes: int = 2
    positional_encoding: str = "sinusoidal"
    pooling: str = "masked_mean"

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.max_sequence_length <= 0:
            raise ValueError(
                f"max_sequence_length must be positive, got {self.max_sequence_length}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.positional_encoding != "sinusoidal":
            raise ValueError(f"Unsupported positional_encoding: {self.positional_encoding!r}")
        if self.pooling != "masked_mean":
            raise ValueError(f"Unsupported pooling: {self.pooling!r}")

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransformerConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
