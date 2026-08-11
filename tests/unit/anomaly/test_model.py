import torch

from evat.anomaly.model import anomaly_map, fit_category_anomaly_model, image_anomaly_score
from evat.features.encoders import CNNEncoderConfig, CNNFeatureEncoder


def _encoder() -> CNNFeatureEncoder:
    return CNNFeatureEncoder(CNNEncoderConfig(pretrained=False, input_height=32, input_width=32))


def test_fit_category_anomaly_model_produces_valid_distributions():
    encoder = _encoder()
    normal_images = torch.randn(6, 3, 32, 32)

    model = fit_category_anomaly_model("bottle", encoder, normal_images)

    assert model.category == "bottle"
    assert model.image_level.feature_dim == encoder.feature_dim
    assert model.pixel_level.feature_dim > 0


def test_image_anomaly_score_shape_and_nonnegative():
    encoder = _encoder()
    normal_images = torch.randn(6, 3, 32, 32)
    model = fit_category_anomaly_model("bottle", encoder, normal_images)

    scores = image_anomaly_score(model, encoder, torch.randn(3, 3, 32, 32))

    assert scores.shape == (3,)
    assert (scores >= 0).all()


def test_normal_like_images_score_lower_than_far_outliers():
    torch.manual_seed(0)
    encoder = _encoder()
    normal_images = torch.zeros(8, 3, 32, 32) + 0.01 * torch.randn(8, 3, 32, 32)
    model = fit_category_anomaly_model("bottle", encoder, normal_images)

    normal_like = torch.zeros(2, 3, 32, 32) + 0.01 * torch.randn(2, 3, 32, 32)
    far_outlier = torch.ones(2, 3, 32, 32) * 50.0

    normal_scores = image_anomaly_score(model, encoder, normal_like)
    outlier_scores = image_anomaly_score(model, encoder, far_outlier)

    assert outlier_scores.mean() > normal_scores.mean()


def test_anomaly_map_shape_matches_feature_grid():
    encoder = _encoder()
    normal_images = torch.randn(6, 3, 32, 32)
    model = fit_category_anomaly_model("bottle", encoder, normal_images)

    amap = anomaly_map(model, encoder, torch.randn(1, 3, 32, 32))

    assert amap.ndim == 2
    assert (amap >= 0).all()


def test_category_isolation_between_models():
    """A category-B-fitted model must not be silently used to score category A,
    and each category's statistics must come only from its own images."""
    torch.manual_seed(0)
    encoder = _encoder()

    category_a_images = torch.zeros(6, 3, 32, 32)
    category_b_images = torch.ones(6, 3, 32, 32) * 5.0

    model_a = fit_category_anomaly_model("category_a", encoder, category_a_images)
    model_b = fit_category_anomaly_model("category_b", encoder, category_b_images)

    assert model_a.category != model_b.category
    assert not (model_a.image_level.mean == model_b.image_level.mean).all()

    # Scoring category A's normal images against B's model should look far more
    # anomalous than scoring them against A's own model.
    score_against_own = image_anomaly_score(model_a, encoder, category_a_images)
    score_against_other = image_anomaly_score(model_b, encoder, category_a_images)

    assert score_against_other.mean() > score_against_own.mean()


def test_fitting_is_deterministic_given_same_inputs():
    encoder = _encoder()
    normal_images = torch.randn(6, 3, 32, 32)

    model_1 = fit_category_anomaly_model("bottle", encoder, normal_images)
    model_2 = fit_category_anomaly_model("bottle", encoder, normal_images)

    assert (model_1.image_level.mean == model_2.image_level.mean).all()
    assert (model_1.image_level.precision == model_2.image_level.precision).all()
