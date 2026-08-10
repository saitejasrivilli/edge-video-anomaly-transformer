import numpy as np
import torch

from evat.features.encoders import BaselineStatsEncoder, CNNEncoderConfig, CNNFeatureEncoder


def _crop(value: int, shape=(8, 8, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_baseline_single_image_to_feature():
    encoder = BaselineStatsEncoder()

    feature = encoder.extract(_crop(100))

    assert feature.shape == (BaselineStatsEncoder.feature_dim,)
    assert feature.dtype == np.float32


def test_baseline_batch_images_to_bd():
    encoder = BaselineStatsEncoder()
    crops = [_crop(10), _crop(200), _crop(50)]

    features = encoder.extract_batch(crops)

    assert features.shape == (3, BaselineStatsEncoder.feature_dim)


def test_baseline_extraction_is_deterministic():
    encoder = BaselineStatsEncoder()
    crop = _crop(77)

    a = encoder.extract(crop)
    b = encoder.extract(crop)

    assert np.array_equal(a, b)


def test_baseline_mask_aware_ignores_background():
    encoder = BaselineStatsEncoder()
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[:2, :2] = 200  # foreground quadrant
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1

    with_mask = encoder.extract(crop, mask=mask)
    without_mask = encoder.extract(crop, mask=None)

    assert with_mask[0] > without_mask[0]  # mean R higher when background excluded


def test_cnn_encoder_constructs_without_downloading_weights():
    config = CNNEncoderConfig(pretrained=False, feature_dim=32)
    encoder = CNNFeatureEncoder(config)

    x = torch.randn(2, 3, config.input_height, config.input_width)
    y = encoder(x)

    assert y.shape == (2, 32)


def test_cnn_encoder_native_dim_when_feature_dim_unset():
    config = CNNEncoderConfig(pretrained=False, feature_dim=None)
    encoder = CNNFeatureEncoder(config)

    x = torch.randn(1, 3, config.input_height, config.input_width)
    y = encoder(x)

    assert y.shape == (1, encoder.feature_dim)


def test_cnn_encoder_frozen_backbone_has_no_grad_params():
    config = CNNEncoderConfig(pretrained=False, frozen=True)
    encoder = CNNFeatureEncoder(config)

    assert all(not p.requires_grad for p in encoder.features.parameters())
