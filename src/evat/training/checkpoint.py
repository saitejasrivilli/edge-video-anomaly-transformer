"""Checkpointing — required because Colab sessions can terminate mid-training.

A checkpoint captures everything needed to resume: model/optimizer/
scheduler state, epoch, the training config, and the metrics recorded at
save time. Checkpoints are never committed to Git (see .gitignore).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from evat.training.config import SegmentationTrainingConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: SegmentationTrainingConfig,
    metrics: dict[str, float],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": asdict(config),
        "metrics": metrics,
        "rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint in-place into ``model`` (and optimizer/scheduler if given).

    Returns the raw payload dict (epoch, config, metrics, rng_state) for
    the caller to resume training from.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload
