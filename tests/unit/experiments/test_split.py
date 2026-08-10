from evat.data.datasets.youtube_vos import ObjectMeta, VideoRecord
from evat.experiments.task_category import build_category_label_map, split_videos_by_video_id


def _video(video_id: str, category: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        split="train",
        objects=(ObjectMeta(object_id="1", category=category, frame_ids=("0",)),),
        frames=(),
    )


def test_build_category_label_map_is_sorted_and_deterministic():
    videos = [_video("a", "dog"), _video("b", "cat"), _video("c", "dog")]

    label_map = build_category_label_map(videos)

    assert label_map == {"cat": 0, "dog": 1}


def test_split_by_video_id_has_no_overlap():
    videos = [_video(f"v{i}", "dog") for i in range(10)]

    train, val = split_videos_by_video_id(videos, val_fraction=0.3, seed=0)

    train_ids = {v.video_id for v in train}
    val_ids = {v.video_id for v in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {v.video_id for v in videos}


def test_split_is_deterministic_given_same_seed():
    videos = [_video(f"v{i}", "dog") for i in range(10)]

    train_a, val_a = split_videos_by_video_id(videos, val_fraction=0.3, seed=7)
    train_b, val_b = split_videos_by_video_id(videos, val_fraction=0.3, seed=7)

    assert [v.video_id for v in train_a] == [v.video_id for v in train_b]
    assert [v.video_id for v in val_a] == [v.video_id for v in val_b]


def test_split_rejects_invalid_val_fraction():
    videos = [_video("a", "dog")]

    for bad_fraction in (0.0, 1.0, -0.1, 1.5):
        try:
            split_videos_by_video_id(videos, val_fraction=bad_fraction, seed=0)
            raise AssertionError(f"expected ValueError for {bad_fraction}")
        except ValueError:
            pass


def test_split_always_reserves_at_least_one_validation_video():
    videos = [_video(f"v{i}", "dog") for i in range(3)]

    _, val = split_videos_by_video_id(videos, val_fraction=0.1, seed=0)

    assert len(val) >= 1
