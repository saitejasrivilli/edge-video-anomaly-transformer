"""End-to-end Phase 2 pipeline test against tiny local fixtures.

YouTube-VOS -> frame sampling -> temporal sequence -> [T, C, H, W] tensors,
with object/mask identity preserved throughout. No real YouTube-VOS data
is downloaded or required.
"""

from pathlib import Path

import pytest

from evat.data.datasets.youtube_vos import build_video_index
from evat.video.sampling import uniform_frame_indices
from evat.video.sequence import build_temporal_sequence
from evat.video.tensors import load_temporal_sequence

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "youtube_vos_mini"


def test_full_pipeline_preserves_identity_and_shape():
    videos = build_video_index(FIXTURE_ROOT, split="train")
    video = videos[0]

    indices = uniform_frame_indices(num_frames_total=len(video.frames), num_samples=2)
    sequence = build_temporal_sequence(video, indices)

    assert sequence.object_ids == ("1",)

    batch = load_temporal_sequence(sequence, dataset_root=FIXTURE_ROOT)

    t, c, h, w = batch.images.shape
    assert t == 2
    assert c == 3
    assert (h, w) == (2, 2)
    assert batch.object_ids[0] == ("1",)
    # First sampled frame (index 0) carries the annotation; later ones may not.
    assert batch.masks[0] is not None
    assert batch.masks[0].tolist() == [[0, 1], [1, 0]]


def test_sequence_rejects_out_of_range_index():
    videos = build_video_index(FIXTURE_ROOT, split="train")
    video = videos[0]

    with pytest.raises(ValueError, match="out of range"):
        build_temporal_sequence(video, [0, 999])
