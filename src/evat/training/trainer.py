"""Training loop.

Separates concerns: this module only orchestrates (optimizer step, epoch
loop, periodic checkpointing/eval); the dataset, model, and loss are
injected, not constructed here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evat.evaluation.evaluator import evaluate
from evat.evaluation.metrics import SegmentationMetrics
from evat.training.checkpoint import save_checkpoint
from evat.training.config import SegmentationTrainingConfig
from evat.training.dataset import SegmentationBatch


def build_optimizer(
    model: torch.nn.Module, config: SegmentationTrainingConfig
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")


def train_step(
    model: torch.nn.Module,
    batch: SegmentationBatch,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Run one optimization step on a single batch. Returns the scalar loss."""
    model.train()
    images = batch.image.to(device)
    targets = batch.mask.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()

    return loss.item()


class Trainer:
    """Owns the epoch loop: train steps, periodic evaluation, periodic checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        config: SegmentationTrainingConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.optimizer = build_optimizer(model, config)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, SegmentationMetrics]:
        """Run ``config.epochs`` epochs. Returns metrics from each evaluated epoch."""
        torch.manual_seed(self.config.seed)
        eval_history: dict[str, SegmentationMetrics] = {}

        for epoch in range(1, self.config.epochs + 1):
            epoch_losses = []
            for batch in train_loader:
                loss = train_step(self.model, batch, self.loss_fn, self.optimizer, self.device)
                epoch_losses.append(loss)

            should_eval = val_loader is not None and epoch % self.config.eval_every == 0
            if should_eval:
                assert val_loader is not None
                metrics = evaluate(self.model, val_loader, self.device)
                eval_history[f"epoch_{epoch}"] = metrics

                checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch}.pt"
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    config=self.config,
                    metrics=asdict(metrics),
                )

        return eval_history
