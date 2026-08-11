import json

from evat.anomaly.metrics import AnomalyMetrics
from evat.anomaly.record import save_anomaly_result


def test_save_anomaly_result_writes_expected_files(tmp_path):
    metrics = AnomalyMetrics(roc_auc=0.8, pr_auc=0.7)

    experiment_dir = save_anomaly_result(
        tmp_path,
        "bottle",
        config={"backbone": "mobilenet_v3_small"},
        metrics=metrics,
        git_commit="abc123",
        dataset_version="fixture-v0",
        hardware="CPU (local smoke test)",
        runtime_seconds=1.5,
        pixel_roc_auc=0.65,
    )

    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "summary.md").exists()

    payload = json.loads((experiment_dir / "metrics.json").read_text())
    assert payload["image_level"]["roc_auc"] == 0.8
    assert payload["pixel_level"]["roc_auc"] == 0.65
