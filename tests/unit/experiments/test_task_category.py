from pathlib import Path

from evat.data.datasets.youtube_vos import build_video_index
from evat.experiments.task_category import build_category_label_map, build_category_samples
from evat.features.encoders import BaselineStatsEncoder

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures" / "youtube_vos_tracking_mini"


def test_build_category_samples_attaches_correct_label():
    videos = build_video_index(FIXTURE_ROOT, split="train")
    label_map = build_category_label_map(videos)

    samples = build_category_samples(
        videos,
        dataset_root=str(FIXTURE_ROOT),
        label_map=label_map,
        encoder=BaselineStatsEncoder(),
        num_frames_per_video=3,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.category == "dog"
    assert sample.label == label_map["dog"]
    assert sample.video_id == "dog"
    assert sample.sequence.length == 3


def test_build_category_samples_skips_categories_not_in_label_map():
    videos = build_video_index(FIXTURE_ROOT, split="train")

    samples = build_category_samples(
        videos,
        dataset_root=str(FIXTURE_ROOT),
        label_map={"some_other_category": 0},  # "dog" deliberately excluded
        encoder=BaselineStatsEncoder(),
        num_frames_per_video=3,
    )

    assert samples == []


def test_build_category_samples_input_never_contains_the_label():
    """The model input (feature vectors) must not encode the category string."""
    videos = build_video_index(FIXTURE_ROOT, split="train")
    label_map = build_category_label_map(videos)

    samples = build_category_samples(
        videos,
        dataset_root=str(FIXTURE_ROOT),
        label_map=label_map,
        encoder=BaselineStatsEncoder(),
        num_frames_per_video=3,
    )

    for sample in samples:
        assert sample.sequence.features.dtype.kind == "f"  # numeric features only
