"""Training configuration.

A single typed, YAML-serializable config controls dataset, model, and
optimization choices so experiments are reproducible without hard-coded
values (CLAUDE.md Section 26).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class SegmentationTrainingConfig:
    """All knobs needed to reproduce a Phase 3 segmentation training run."""

    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 1
    optimizer: str = "adam"
    scheduler: str | None = None
    input_height: int = 128
    input_width: int = 128
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 16
    depth: int = 3
    checkpoint_dir: str = "checkpoints/segmentation"
    eval_every: int = 1

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SegmentationTrainingConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
