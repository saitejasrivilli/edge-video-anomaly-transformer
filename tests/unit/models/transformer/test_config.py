import pytest

from evat.models.transformer.config import TransformerConfig


def test_config_roundtrips_through_yaml(tmp_path):
    config = TransformerConfig(d_model=32, num_heads=4)
    path = tmp_path / "transformer.yaml"

    config.to_yaml(path)
    loaded = TransformerConfig.from_yaml(path)

    assert loaded == config


def test_rejects_d_model_not_divisible_by_num_heads():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(d_model=10, num_heads=3)


def test_rejects_invalid_max_sequence_length():
    with pytest.raises(ValueError, match="max_sequence_length"):
        TransformerConfig(max_sequence_length=0)


def test_rejects_invalid_feature_dim():
    with pytest.raises(ValueError, match="feature_dim"):
        TransformerConfig(feature_dim=0)


def test_rejects_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes"):
        TransformerConfig(num_classes=0)


def test_rejects_invalid_num_layers():
    with pytest.raises(ValueError, match="num_layers"):
        TransformerConfig(num_layers=0)
