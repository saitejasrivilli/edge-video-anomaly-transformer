"""Temporal sequence construction.

Builds a ``TemporalSequence`` — an ordered subset of a video's frames,
selected by index — while preserving each frame's annotation path and
object-ID metadata. No pixel data is touched here; see ``tensors.py`` for
loading actual image/mask content.
"""

from __future__ import annotations

from dataclasses import dataclass

from evat.data.datasets.youtube_vos import FrameRecord, VideoRecord


@dataclass(frozen=True, slots=True)
class TemporalSequence:
    """An ordered sequence of frames sampled from one video."""

    video_id: str
    split: str
    frame_indices: tuple[int, ...]
    frames: tuple[FrameRecord, ...]

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Union of object IDs present across all frames in this sequence, sorted."""
        ids: set[str] = set()
        for frame in self.frames:
            ids.update(frame.object_ids)
        return tuple(sorted(ids))


def build_temporal_sequence(video: VideoRecord, frame_indices: list[int]) -> TemporalSequence:
    """Select frames from ``video`` at ``frame_indices``, preserving order.

    Raises:
        ValueError: if any index is out of range, or ``frame_indices`` is empty.
    """
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")

    total = len(video.frames)
    for idx in frame_indices:
        if not 0 <= idx < total:
            raise ValueError(
                f"frame index {idx} out of range for video '{video.video_id}' with {total} frames"
            )

    selected = tuple(video.frames[idx] for idx in frame_indices)
    return TemporalSequence(
        video_id=video.video_id,
        split=video.split,
        frame_indices=tuple(frame_indices),
        frames=selected,
    )
