"""End-to-end Phase 5 pipeline test, reusing Phase 2 (temporal reader) and
Phase 4 (tracker) infrastructure unchanged.

YouTube-VOS -> temporal sequence -> ground-truth-derived candidates ->
tracker -> object crops -> baseline features -> temporal feature sequence.

No real YouTube-VOS data or pretrained weights are downloaded/required.
"""

from pathlib import Path

from evat.data.datasets.youtube_vos import build_video_index
from evat.features.encoders import BaselineStatsEncoder
from evat.features.extract import extract_object_features_baseline
from evat.features.temporal import build_temporal_feature_sequence, group_features_by_track
from evat.tracking.ground_truth import extract_ground_truth_instances, strip_identity
from evat.tracking.tracker import Tracker
from evat.video.sampling import uniform_frame_indices
from evat.video.sequence import build_temporal_sequence
from evat.video.tensors import load_temporal_sequence

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "youtube_vos_tracking_mini"


def test_features_flow_from_tracked_instances_to_temporal_sequence():
    videos = build_video_index(FIXTURE_ROOT, split="train")
    video = videos[0]
    indices = uniform_frame_indices(num_frames_total=len(video.frames), num_samples=3)
    sequence = build_temporal_sequence(video, indices)
    batch = load_temporal_sequence(sequence, dataset_root=FIXTURE_ROOT)

    tracker = Tracker()
    encoder = BaselineStatsEncoder()
    all_features = []
    frame_order = list(batch.frame_ids)

    for i, frame_id in enumerate(frame_order):
        object_id_mask = batch.masks[i]
        assert object_id_mask is not None  # this fixture annotates every frame
        gt_instances = extract_ground_truth_instances(object_id_mask, frame_id)
        candidates = strip_identity(gt_instances)
        tracked = tracker.update(frame_id, candidates)

        frame_rgb = batch.images[i].transpose(1, 2, 0)
        all_features.extend(extract_object_features_baseline(frame_rgb, tracked, encoder))

    grouped = group_features_by_track(all_features)
    assert len(grouped) == 1  # single ground-truth object across all frames

    track_id = next(iter(grouped))
    temporal_sequence = build_temporal_feature_sequence(track_id, frame_order, grouped[track_id])

    assert temporal_sequence.length == 3
    assert temporal_sequence.validity.all()
    assert temporal_sequence.feature_dim == BaselineStatsEncoder.feature_dim
