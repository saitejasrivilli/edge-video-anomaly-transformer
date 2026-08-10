"""YouTube-VOS dataset adapter.

License: annotations CC BY 4.0; dataset as a whole restricted to
non-commercial research use per the official Terms of Use. See
docs/datasets.md ("Selected Video Dataset") for the full verified record.
This module locates, parses, and indexes a YouTube-VOS split that the user
has already downloaded through the official channel — it never downloads
data itself.

Expected on-disk structure (as distributed by YouTube-VOS), per split
(e.g. "train"):

    <root>/<split>/meta.json
    <root>/<split>/JPEGImages/<video_id>/<frame_id>.jpg
    <root>/<split>/Annotations/<video_id>/<frame_id>.png

``meta.json`` maps each video to its objects:

    {"videos": {"<video_id>": {"objects": {"<object_id>": {"category": "...",
                                                             "frames": ["<frame_id>", ...]}}}}}

Annotation PNGs are palette-indexed: pixel values equal object IDs. Object
identity is never discarded — it is threaded through ``VideoRecord`` and
``FrameRecord`` so the later tracking phase can consume it directly.

No image resizing, normalization, or augmentation happens here — that is
out of scope for Phase 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evat.data.validation import ValidationIssue, ValidationReport

DATASET_NAME = "youtube_vos"


@dataclass(frozen=True, slots=True)
class ObjectMeta:
    """One tracked object within a video, as declared in meta.json."""

    object_id: str
    category: str
    frame_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One frame of one video, with its annotation and the object IDs present."""

    video_id: str
    frame_id: str
    image_path: str
    annotation_path: str | None
    object_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """A full video: its ordered frames and its object metadata."""

    video_id: str
    split: str
    objects: tuple[ObjectMeta, ...]
    frames: tuple[FrameRecord, ...]


def parse_meta(meta_path: Path) -> dict[str, tuple[ObjectMeta, ...]]:
    """Parse meta.json into per-video object metadata.

    Raises:
        ValueError: if the file is missing or malformed.
    """
    if not meta_path.is_file():
        raise ValueError(f"YouTube-VOS meta.json not found at: {meta_path}")

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed meta.json at '{meta_path}': {exc}") from exc

    videos = data.get("videos")
    if not isinstance(videos, dict):
        raise ValueError(f"meta.json at '{meta_path}' is missing a 'videos' mapping")

    result: dict[str, tuple[ObjectMeta, ...]] = {}
    for video_id, video_data in videos.items():
        objects_data = video_data.get("objects", {})
        objects = tuple(
            ObjectMeta(
                object_id=object_id,
                category=obj["category"],
                frame_ids=tuple(obj["frames"]),
            )
            for object_id, obj in objects_data.items()
        )
        result[video_id] = objects

    return result


def discover_videos(root: Path, split: str) -> list[str]:
    """List video IDs present under <root>/<split>/JPEGImages."""
    jpeg_dir = root / split / "JPEGImages"
    if not jpeg_dir.is_dir():
        raise ValueError(f"JPEGImages directory not found: {jpeg_dir}")
    return sorted(p.name for p in jpeg_dir.iterdir() if p.is_dir())


def build_video_index(root: Path, split: str) -> list[VideoRecord]:
    """Scan a YouTube-VOS split and build one VideoRecord per video.

    Does not read image/mask pixel content — only filesystem structure and
    meta.json metadata. Frame order follows sorted frame-file order within
    each video's JPEGImages directory (YouTube-VOS frame IDs are
    zero-padded and sort correctly as strings).
    """
    split_dir = root / split
    meta = parse_meta(split_dir / "meta.json")
    annotations_dir = split_dir / "Annotations"

    records: list[VideoRecord] = []
    for video_id in discover_videos(root, split):
        video_objects = meta.get(video_id, ())
        frame_to_objects: dict[str, list[str]] = {}
        for obj in video_objects:
            for frame_id in obj.frame_ids:
                frame_to_objects.setdefault(frame_id, []).append(obj.object_id)

        jpeg_dir = split_dir / "JPEGImages" / video_id
        frame_paths = sorted(jpeg_dir.glob("*.jpg"))

        frames = []
        for image_path in frame_paths:
            frame_id = image_path.stem
            annotation_path = annotations_dir / video_id / f"{frame_id}.png"
            frames.append(
                FrameRecord(
                    video_id=video_id,
                    frame_id=frame_id,
                    image_path=str(image_path.relative_to(root)),
                    annotation_path=(
                        str(annotation_path.relative_to(root)) if annotation_path.exists() else None
                    ),
                    object_ids=tuple(frame_to_objects.get(frame_id, ())),
                )
            )

        records.append(
            VideoRecord(
                video_id=video_id,
                split=split,
                objects=video_objects,
                frames=tuple(frames),
            )
        )

    return records


def validate_video_index(records: list[VideoRecord]) -> ValidationReport:
    """Validate a list of VideoRecords for structural and referential integrity.

    Checks performed:
    - duplicate video IDs
    - videos with zero frames
    - object frame_ids referenced in meta.json that don't exist among the
      video's discovered frames (dangling metadata reference)

    Frames with no annotation are not treated as errors: YouTube-VOS's
    semi-supervised protocol only guarantees a first-frame annotation for
    val/test splits, so intermediate unannotated frames are expected.
    """
    issues: list[ValidationIssue] = []
    seen_video_ids: set[str] = set()

    for record in records:
        if record.video_id in seen_video_ids:
            issues.append(ValidationIssue(record.video_id, "duplicate video_id"))
        seen_video_ids.add(record.video_id)

        if not record.frames:
            issues.append(ValidationIssue(record.video_id, "video has zero frames"))
            continue

        known_frame_ids = {f.frame_id for f in record.frames}
        for obj in record.objects:
            dangling = set(obj.frame_ids) - known_frame_ids
            for frame_id in sorted(dangling):
                issues.append(
                    ValidationIssue(
                        record.video_id,
                        f"object '{obj.object_id}' references frame_id '{frame_id}' "
                        "not found among video frames",
                    )
                )

    return ValidationReport(total_records=len(records), issues=tuple(issues))
