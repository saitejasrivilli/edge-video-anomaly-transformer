"""Experiment recording: writes config.yaml, metrics.json, summary.md.

Called from Colab after a real training/evaluation run — never invents
numbers. See CLAUDE.md Section 31 (Experiment Tracking).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evat.evaluation.metrics import SegmentationMetrics
from evat.training.config import SegmentationTrainingConfig


def save_experiment_record(
    results_dir: str | Path,
    experiment_name: str,
    config: SegmentationTrainingConfig,
    metrics: SegmentationMetrics,
    git_commit: str,
    dataset_version: str,
    hardware: str,
    runtime_seconds: float,
) -> Path:
    """Write ``results/segmentation/<experiment_name>/{config.yaml,metrics.json,summary.md}``."""
    experiment_dir = Path(results_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config.to_yaml(experiment_dir / "config.yaml")

    metrics_payload = {
        "metrics": asdict(metrics),
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "hardware": hardware,
        "runtime_seconds": runtime_seconds,
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = (
        f"# {experiment_name}\n\n"
        f"- Git commit: `{git_commit}`\n"
        f"- Dataset version: `{dataset_version}`\n"
        f"- Hardware: {hardware}\n"
        f"- Runtime: {runtime_seconds:.1f}s\n\n"
        f"## Metrics (pixel-level)\n\n"
        f"- IoU: {metrics.iou:.4f}\n"
        f"- Dice: {metrics.dice:.4f}\n"
        f"- Precision: {metrics.precision:.4f}\n"
        f"- Recall: {metrics.recall:.4f}\n"
    )
    (experiment_dir / "summary.md").write_text(summary, encoding="utf-8")

    return experiment_dir
