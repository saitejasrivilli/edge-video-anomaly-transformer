"""Experiment recording for Phase 7 — same config.yaml/metrics.json/summary.md
convention as segmentation (Phase 3) and used for reproducibility (CLAUDE.md
Section 22): every reported result must trace to a real run's Git commit,
config, and dataset version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evat.experiments.metrics import ClassificationMetrics


def save_experiment_result(
    results_dir: str | Path,
    experiment_name: str,
    config: dict[str, Any],
    metrics: ClassificationMetrics,
    git_commit: str,
    dataset_version: str,
    hardware: str,
    runtime_seconds: float,
) -> Path:
    """Write ``results/phase7/<experiment_name>/{config.json,metrics.json,summary.md}``."""
    experiment_dir = Path(results_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    metrics_payload = {
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
        },
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
        f"## Metrics\n\n"
        f"- Accuracy: {metrics.accuracy:.4f}\n"
        f"- Macro F1: {metrics.macro_f1:.4f}\n"
        f"- Macro Precision: {metrics.macro_precision:.4f}\n"
        f"- Macro Recall: {metrics.macro_recall:.4f}\n"
    )
    (experiment_dir / "summary.md").write_text(summary, encoding="utf-8")

    return experiment_dir
