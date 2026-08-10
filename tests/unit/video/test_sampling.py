import pytest

from evat.video.sampling import strided_frame_indices, uniform_frame_indices


def test_uniform_frame_indices_includes_first_frame():
    indices = uniform_frame_indices(num_frames_total=10, num_samples=4)
    assert indices[0] == 0
    assert len(indices) == 4
    assert indices == sorted(indices)


def test_uniform_frame_indices_within_bounds():
    indices = uniform_frame_indices(num_frames_total=100, num_samples=8)
    assert all(0 <= i < 100 for i in indices)
    assert len(set(indices)) == 8


def test_uniform_frame_indices_more_samples_than_frames_returns_all():
    indices = uniform_frame_indices(num_frames_total=3, num_samples=10)
    assert indices == [0, 1, 2]


@pytest.mark.parametrize("total,samples", [(0, 4), (-1, 4), (10, 0), (10, -1)])
def test_uniform_frame_indices_rejects_non_positive(total, samples):
    with pytest.raises(ValueError):
        uniform_frame_indices(total, samples)


def test_strided_frame_indices():
    assert strided_frame_indices(num_frames_total=10, stride=3) == [0, 3, 6, 9]


def test_strided_frame_indices_rejects_invalid_stride():
    with pytest.raises(ValueError):
        strided_frame_indices(num_frames_total=10, stride=0)
