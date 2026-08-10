import json

from evat.evaluation.experiment import save_experiment_record
from evat.evaluation.metrics import SegmentationMetrics
from evat.training.config import SegmentationTrainingConfig


def test_save_experiment_record_writes_expected_files(tmp_path):
    config = SegmentationTrainingConfig()
    metrics = SegmentationMetrics(iou=0.1, dice=0.2, precision=0.3, recall=0.4)

    experiment_dir = save_experiment_record(
        tmp_path,
        "smoke_test",
        config=config,
        metrics=metrics,
        git_commit="abc123",
        dataset_version="fixture-v0",
        hardware="CPU (local smoke test)",
        runtime_seconds=1.23,
    )

    assert (experiment_dir / "config.yaml").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "summary.md").exists()

    payload = json.loads((experiment_dir / "metrics.json").read_text())
    assert payload["metrics"]["iou"] == 0.1
    assert payload["git_commit"] == "abc123"
