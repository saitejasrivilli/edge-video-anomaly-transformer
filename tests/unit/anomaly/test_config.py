from evat.anomaly.config import AnomalyConfig


def test_config_roundtrips_through_yaml(tmp_path):
    config = AnomalyConfig(threshold_percentile=90.0, covariance_eps=1e-2)
    path = tmp_path / "anomaly.yaml"

    config.to_yaml(path)
    loaded = AnomalyConfig.from_yaml(path)

    assert loaded == config
