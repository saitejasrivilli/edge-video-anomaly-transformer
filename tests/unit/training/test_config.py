from evat.training.config import SegmentationTrainingConfig


def test_config_roundtrips_through_yaml(tmp_path):
    config = SegmentationTrainingConfig(seed=123, batch_size=2, epochs=3)
    path = tmp_path / "config.yaml"

    config.to_yaml(path)
    loaded = SegmentationTrainingConfig.from_yaml(path)

    assert loaded == config


def test_config_defaults_are_reasonable_for_local_smoke_tests():
    config = SegmentationTrainingConfig()

    assert config.batch_size > 0
    assert config.epochs > 0
    assert config.input_height > 0
    assert config.input_width > 0
