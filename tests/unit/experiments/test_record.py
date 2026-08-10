import json

from evat.experiments.metrics import ClassificationMetrics
from evat.experiments.record import save_experiment_result


def test_save_experiment_result_writes_expected_files(tmp_path):
    metrics = ClassificationMetrics(
        accuracy=0.5, macro_f1=0.4, macro_precision=0.45, macro_recall=0.42
    )

    experiment_dir = save_experiment_result(
        tmp_path,
        "baseline_mlp",
        config={"batch_size": 8},
        metrics=metrics,
        git_commit="abc123",
        dataset_version="fixture-v0",
        hardware="CPU (local smoke test)",
        runtime_seconds=2.5,
    )

    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "summary.md").exists()

    payload = json.loads((experiment_dir / "metrics.json").read_text())
    assert payload["metrics"]["macro_f1"] == 0.4
    assert payload["git_commit"] == "abc123"
