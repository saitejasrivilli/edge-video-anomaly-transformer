"""Load a TemporalSequence's pixel data into arrays.

Images are stacked into a ``[T, C, H, W]`` uint8 array, exactly as decoded —
no resizing, normalization, or augmentation (out of scope for Phase 2).
Annotation masks stay palette-indexed (pixel value == object ID) and are
kept as a per-frame list, since a frame may have no annotation at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evat.video.sequence import TemporalSequence


@dataclass(frozen=True)
class TemporalTensorBatch:
    """Loaded pixel data + preserved identity metadata for one TemporalSequence."""

    video_id: str
    frame_ids: tuple[str, ...]
    images: np.ndarray  # [T, C, H, W], uint8
    masks: tuple[np.ndarray | None, ...]  # per-frame, [H, W] uint8 palette index or None
    object_ids: tuple[tuple[str, ...], ...]  # per-frame object IDs present


def load_temporal_sequence(
    sequence: TemporalSequence, dataset_root: str | Path
) -> TemporalTensorBatch:
    """Decode a TemporalSequence's images and masks relative to ``dataset_root``.

    Raises:
        ValueError: if referenced image files are missing, or frames have
            inconsistent dimensions (which would make stacking into a
            single [T, C, H, W] array meaningless).
    """
    dataset_root = Path(dataset_root)

    images: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []

    for frame in sequence.frames:
        image_path = dataset_root / frame.image_path
        if not image_path.is_file():
            raise ValueError(f"Frame image not found: {image_path}")

        with Image.open(image_path) as img:
            array = np.array(img.convert("RGB"))  # [H, W, C]
        images.append(np.transpose(array, (2, 0, 1)))  # -> [C, H, W]

        if frame.annotation_path is not None:
            mask_path = dataset_root / frame.annotation_path
            if not mask_path.is_file():
                raise ValueError(f"Annotation mask not found: {mask_path}")
            with Image.open(mask_path) as mask_img:
                masks.append(np.array(mask_img))  # palette index values preserved
        else:
            masks.append(None)

    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent frame shapes in video '{sequence.video_id}': {shapes}. "
            "Cannot stack into a single [T, C, H, W] array."
        )

    return TemporalTensorBatch(
        video_id=sequence.video_id,
        frame_ids=tuple(f.frame_id for f in sequence.frames),
        images=np.stack(images, axis=0),
        masks=tuple(masks),
        object_ids=tuple(f.object_ids for f in sequence.frames),
    )
