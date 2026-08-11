"""Experiment recording for Phase 8 — same config/metrics/summary convention
used since Phase 3 (CLAUDE.md Section 18: every result must trace to a
Git commit, config, and dataset version).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evat.anomaly.metrics import AnomalyMetrics


def save_anomaly_result(
    results_dir: str | Path,
    category: str,
    config: dict[str, Any],
    metrics: AnomalyMetrics,
    git_commit: str,
    dataset_version: str,
    hardware: str,
    runtime_seconds: float,
    pixel_roc_auc: float | None = None,
) -> Path:
    """Write ``results/phase8/<category>/{config.json,metrics.json,summary.md}``."""
    experiment_dir = Path(results_dir) / category
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    metrics_payload = {
        "image_level": {"roc_auc": metrics.roc_auc, "pr_auc": metrics.pr_auc},
        "pixel_level": {"roc_auc": pixel_roc_auc},
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "hardware": hardware,
        "runtime_seconds": runtime_seconds,
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = (
        f"# {category}\n\n"
        f"- Git commit: `{git_commit}`\n"
        f"- Dataset version: `{dataset_version}`\n"
        f"- Hardware: {hardware}\n"
        f"- Runtime: {runtime_seconds:.1f}s\n\n"
        f"## Image-level metrics\n\n"
        f"- ROC-AUC: {metrics.roc_auc:.4f}\n"
        f"- PR-AUC: {metrics.pr_auc:.4f}\n\n"
        f"## Pixel-level metrics\n\n"
        f"- ROC-AUC: {f'{pixel_roc_auc:.4f}' if pixel_roc_auc is not None else 'not computed'}\n"
    )
    (experiment_dir / "summary.md").write_text(summary, encoding="utf-8")

    return experiment_dir
