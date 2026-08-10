import numpy as np
import pytest

from evat.features.schemas import VisualFeature
from evat.features.temporal import build_temporal_feature_sequence, group_features_by_track


def _feature(frame_id: str, track_id: int, value: float) -> VisualFeature:
    return VisualFeature(
        frame_id=frame_id,
        track_id=track_id,
        feature=np.array([value, value], dtype=np.float32),
        extractor_name="test",
    )


def test_group_features_by_track_excludes_global_features():
    features = [
        _feature("0", 1, 1.0),
        _feature("1", 1, 2.0),
        VisualFeature(frame_id="0", track_id=None, feature=np.zeros(2), extractor_name="test"),
    ]

    grouped = group_features_by_track(features)

    assert set(grouped.keys()) == {1}
    assert set(grouped[1].keys()) == {"0", "1"}


def test_temporal_ordering_follows_frame_order():
    features_by_frame = {"0": _feature("0", 1, 1.0), "1": _feature("1", 1, 2.0)}

    sequence = build_temporal_feature_sequence(1, ["0", "1"], features_by_frame)

    assert sequence.frame_ids == ("0", "1")
    assert sequence.features[0, 0] == 1.0
    assert sequence.features[1, 0] == 2.0


def test_missing_frame_is_padded_and_marked_invalid():
    features_by_frame = {"0": _feature("0", 1, 1.0), "2": _feature("2", 1, 3.0)}

    sequence = build_temporal_feature_sequence(1, ["0", "1", "2"], features_by_frame)

    assert sequence.validity.tolist() == [True, False, True]
    assert (sequence.features[1] == 0).all()


def test_sequence_length_pads_when_shorter_than_available_frames():
    features_by_frame = {"0": _feature("0", 1, 1.0)}

    sequence = build_temporal_feature_sequence(1, ["0"], features_by_frame, sequence_length=4)

    assert sequence.length == 4
    assert sequence.frame_ids == ("0", None, None, None)
    assert sequence.validity.tolist() == [True, False, False, False]


def test_sequence_length_truncates_when_longer_than_requested():
    features_by_frame = {"0": _feature("0", 1, 1.0), "1": _feature("1", 1, 2.0)}

    sequence = build_temporal_feature_sequence(1, ["0", "1"], features_by_frame, sequence_length=1)

    assert sequence.length == 1
    assert sequence.frame_ids == ("0",)


def test_stride_samples_every_nth_frame():
    features_by_frame = {"0": _feature("0", 1, 1.0), "2": _feature("2", 1, 3.0)}

    sequence = build_temporal_feature_sequence(1, ["0", "1", "2", "3"], features_by_frame, stride=2)

    assert sequence.frame_ids == ("0", "2")
    assert sequence.validity.tolist() == [True, True]


def test_raises_when_no_features_available_for_track():
    with pytest.raises(ValueError, match="No features available"):
        build_temporal_feature_sequence(1, ["0", "1"], {})
