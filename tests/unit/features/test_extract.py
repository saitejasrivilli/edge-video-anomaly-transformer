import numpy as np

from evat.features.encoders import BaselineStatsEncoder, CNNEncoderConfig, CNNFeatureEncoder
from evat.features.extract import (
    extract_global_feature_baseline,
    extract_object_features_baseline,
    extract_object_features_cnn,
)
from evat.tracking.schemas import TrackedInstance, TrackState


def _frame(shape=(16, 16, 3)) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8)


def _instance(track_id: int, bbox, frame_id="0") -> TrackedInstance:
    mask = np.zeros((16, 16), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    return TrackedInstance(
        frame_id=frame_id, track_id=track_id, state=TrackState.ACTIVE, mask=mask, bbox=bbox
    )


def test_extract_object_features_baseline_associates_track_id():
    frame = _frame()
    instances = [_instance(1, (2, 2, 6, 6)), _instance(2, (10, 10, 14, 14))]

    features = extract_object_features_baseline(frame, instances, BaselineStatsEncoder())

    assert {f.track_id for f in features} == {1, 2}
    assert all(f.frame_id == "0" for f in features)


def test_extract_object_features_baseline_skips_empty_bbox():
    frame = _frame()
    instance = TrackedInstance(
        frame_id="0", track_id=1, state=TrackState.ACTIVE, mask=np.zeros((16, 16)), bbox=None
    )

    features = extract_object_features_baseline(frame, [instance], BaselineStatsEncoder())

    assert features == []


def test_extract_object_features_cnn_produces_correct_dim():
    frame = _frame()
    instances = [_instance(1, (2, 2, 10, 10))]
    encoder = CNNFeatureEncoder(CNNEncoderConfig(pretrained=False, feature_dim=16))

    features = extract_object_features_cnn(frame, instances, encoder)

    assert len(features) == 1
    assert features[0].track_id == 1
    assert features[0].feature_dim == 16


def test_extract_global_feature_has_no_track_id():
    frame = _frame()

    feature = extract_global_feature_baseline(frame, "0", BaselineStatsEncoder())

    assert feature.track_id is None
