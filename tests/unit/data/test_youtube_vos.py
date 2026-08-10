from pathlib import Path

import pytest

from evat.data.datasets.youtube_vos import (
    build_video_index,
    discover_videos,
    parse_meta,
    validate_video_index,
)

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures" / "youtube_vos_mini"


def test_parse_meta_reads_objects():
    meta = parse_meta(FIXTURE_ROOT / "train" / "meta.json")

    assert "bear" in meta
    assert len(meta["bear"]) == 1
    assert meta["bear"][0].category == "bear"
    assert meta["bear"][0].frame_ids == ("00000", "00001", "00002")


def test_parse_meta_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        parse_meta(tmp_path / "meta.json")


def test_parse_meta_raises_on_malformed_json(tmp_path):
    bad = tmp_path / "meta.json"
    bad.write_text("{not valid json")
    with pytest.raises(ValueError, match="Malformed"):
        parse_meta(bad)


def test_discover_videos_finds_fixture_video():
    assert discover_videos(FIXTURE_ROOT, split="train") == ["bear"]


def test_build_video_index_preserves_object_identity():
    records = build_video_index(FIXTURE_ROOT, split="train")

    assert len(records) == 1
    video = records[0]
    assert video.video_id == "bear"
    assert len(video.frames) == 3
    assert video.frames[0].object_ids == ("1",)
    assert video.frames[0].annotation_path == "train/Annotations/bear/00000.png"
    # Only the first frame is annotated (semi-supervised protocol).
    assert video.frames[1].annotation_path is None
    assert video.frames[1].object_ids == ("1",)


def test_build_video_index_passes_validation():
    records = build_video_index(FIXTURE_ROOT, split="train")

    report = validate_video_index(records)

    assert report.is_valid, report.summary()


def test_validate_video_index_detects_duplicate_video_id():
    records = build_video_index(FIXTURE_ROOT, split="train")
    duplicated = records + records

    report = validate_video_index(duplicated)

    assert not report.is_valid
    assert any("duplicate video_id" in str(issue) for issue in report.issues)


def test_validate_video_index_detects_dangling_frame_reference():
    records = build_video_index(FIXTURE_ROOT, split="train")
    video = records[0]
    bad_object = video.objects[0]
    tampered_object = type(bad_object)(
        object_id=bad_object.object_id,
        category=bad_object.category,
        frame_ids=(*bad_object.frame_ids, "99999"),
    )
    tampered_video = type(video)(
        video_id=video.video_id,
        split=video.split,
        objects=(tampered_object,),
        frames=video.frames,
    )

    report = validate_video_index([tampered_video])

    assert not report.is_valid
    assert any("99999" in str(issue) for issue in report.issues)
