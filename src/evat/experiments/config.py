"""Controlled-comparison experiment configuration.

One shared config controls dataset/split/training settings that must be
IDENTICAL across the baseline MLP, GRU baseline, and Transformer for the
comparison to be meaningful (CLAUDE.md Phase 7 Section 9). Model-specific
architecture parameters (e.g. Transformer layers/heads) still come from
each model's own config (``TransformerConfig`` etc.) — this config only
holds what must be shared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(slots=True)
class Phase7ExperimentConfig:
    seed: int = 42
    val_fraction: float = 0.2
    num_frames_per_video: int = 16
    sequence_length: int = 16
    stride: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 20
    optimizer: str = "adam"

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Phase7ExperimentConfig:
        """Load config, ignoring unrelated top-level keys (e.g. ``ablations``)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known_fields})
