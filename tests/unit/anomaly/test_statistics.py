import numpy as np
import pytest

from evat.anomaly.statistics import fit_normal_distribution, mahalanobis_distance


def test_fit_normal_distribution_recovers_mean():
    rng = np.random.default_rng(0)
    features = rng.normal(loc=5.0, scale=1.0, size=(200, 4))

    dist = fit_normal_distribution(features)

    np.testing.assert_allclose(dist.mean, features.mean(axis=0), atol=1e-6)
    assert dist.feature_dim == 4


def test_fit_normal_distribution_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        fit_normal_distribution(np.zeros((1, 4)))


def test_fit_normal_distribution_rejects_wrong_shape():
    with pytest.raises(ValueError, match="N, D"):
        fit_normal_distribution(np.zeros(10))


def test_mahalanobis_distance_is_zero_at_the_mean():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 3))
    dist = fit_normal_distribution(features)

    distance = mahalanobis_distance(dist.mean, dist)

    assert distance == pytest.approx(0.0, abs=1e-6)


def test_mahalanobis_distance_increases_with_deviation():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 3))
    dist = fit_normal_distribution(features)

    near = dist.mean + 0.1
    far = dist.mean + 10.0

    assert mahalanobis_distance(far, dist) > mahalanobis_distance(near, dist)


def test_mahalanobis_distance_batched_shape():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, 3))
    dist = fit_normal_distribution(features)

    distances = mahalanobis_distance(rng.normal(size=(10, 3)), dist)

    assert distances.shape == (10,)
    assert (distances >= 0).all()
