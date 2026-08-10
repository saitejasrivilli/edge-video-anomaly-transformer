"""End-to-end Phase 4 pipeline test: reuses the Phase 2 temporal reader/sequence
abstractions (no separate video reader is built) and feeds ground-truth
object-ID masks through the tracker, evaluating against those same IDs.

YouTube-VOS -> temporal sequence -> object-ID masks -> anonymized candidates
-> tracker -> predicted tracks -> evaluation vs. ground truth.

No real YouTube-VOS data is downloaded or required.
"""

from pathlib import Path

from evat.data.datasets.youtube_vos import build_video_index
from evat.tracking.ground_truth import extract_ground_truth_instances, strip_identity
from evat.tracking.metrics import evaluate_tracking
from evat.tracking.tracker import Tracker
from evat.video.sampling import uniform_frame_indices
from evat.video.sequence import build_temporal_sequence
from evat.video.tensors import load_temporal_sequence

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "youtube_vos_tracking_mini"


def test_tracking_over_a_temporal_sequence_recovers_ground_truth_identity():
    videos = build_video_index(FIXTURE_ROOT, split="train")
    video = videos[0]

    indices = uniform_frame_indices(num_frames_total=len(video.frames), num_samples=3)
    sequence = build_temporal_sequence(video, indices)
    batch = load_temporal_sequence(sequence, dataset_root=FIXTURE_ROOT)

    tracker = Tracker()
    gt_by_frame = {}
    predictions_by_frame = {}
    frame_order = list(batch.frame_ids)

    for frame_id, object_id_mask in zip(batch.frame_ids, batch.masks, strict=True):
        assert object_id_mask is not None  # this fixture annotates every frame
        gt_instances = extract_ground_truth_instances(object_id_mask, frame_id)
        gt_by_frame[frame_id] = gt_instances

        candidates = strip_identity(gt_instances)
        predictions_by_frame[frame_id] = tracker.update(frame_id, candidates)

    metrics = evaluate_tracking(gt_by_frame, predictions_by_frame, frame_order)

    # A single, slowly-moving ground-truth object should be tracked as one
    # consistent identity across all 3 frames.
    assert metrics.coverage == 1.0
    assert metrics.id_consistency == 1.0
    assert metrics.identity_switches == 0
    assert len({t.track_id for outputs in predictions_by_frame.values() for t in outputs}) == 1
