"""Anomaly-detection configuration — nothing hard-coded."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class AnomalyConfig:
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = False  # true only in Colab; local tests always use false (no network)
    frozen: bool = True
    input_height: int = 128
    input_width: int = 128
    covariance_eps: float = 1e-3
    threshold_percentile: float = 95.0

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnomalyConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
