"""Segmentation dataset built on top of the Phase 2 YouTube-VOS data layer.

Reuses ``evat.data.datasets.youtube_vos.VideoRecord``/``FrameRecord``
directly rather than re-parsing YouTube-VOS structure. Only frames that
have an annotation (``annotation_path is not None``) are used as training
samples, since supervised segmentation requires ground truth.

Object representation (baseline decision, see docs/architecture.md):
YouTube-VOS masks are palette-indexed with pixel value == object ID. This
baseline collapses all nonzero object IDs into a single binary
foreground/background target (``mask``), because Phase 3's goal is a
single-object-agnostic segmentation baseline, not per-instance
classification. The original per-pixel object-ID mask is NOT discarded:
it is returned unchanged as ``object_id_mask`` alongside the binary
target, together with the per-frame ``object_ids`` tuple, so Phase 4
(tracking) can consume real object identities rather than reconstructing
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from evat.data.datasets.youtube_vos import FrameRecord, VideoRecord
from evat.training.transforms import horizontal_flip, normalize_image, resize_image, resize_mask


@dataclass(frozen=True, slots=True)
class SegmentationSample:
    """One model-ready segmentation training sample."""

    video_id: str
    frame_id: str
    image: torch.Tensor  # [C, H, W] float, normalized
    mask: torch.Tensor  # [1, H, W] float, binary foreground/background
    object_id_mask: torch.Tensor  # [H, W] long, original per-pixel object IDs
    object_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SegmentationBatch:
    """A collated batch of ``SegmentationSample``s, ready for the model."""

    video_ids: list[str]
    frame_ids: list[str]
    image: torch.Tensor  # [B, C, H, W]
    mask: torch.Tensor  # [B, 1, H, W]
    object_id_mask: torch.Tensor  # [B, H, W]
    object_ids: list[tuple[str, ...]]


def collate_segmentation_batch(samples: list[SegmentationSample]) -> SegmentationBatch:
    """DataLoader ``collate_fn`` for ``SegmentationDataset``.

    Object counts vary per frame, so ``object_ids`` is kept as a plain
    per-sample list rather than stacked into a tensor.
    """
    return SegmentationBatch(
        video_ids=[s.video_id for s in samples],
        frame_ids=[s.frame_id for s in samples],
        image=torch.stack([s.image for s in samples]),
        mask=torch.stack([s.mask for s in samples]),
        object_id_mask=torch.stack([s.object_id_mask for s in samples]),
        object_ids=[s.object_ids for s in samples],
    )


def _annotated_frames(records: list[VideoRecord]) -> list[FrameRecord]:
    return [
        frame for video in records for frame in video.frames if frame.annotation_path is not None
    ]


class SegmentationDataset(Dataset[SegmentationSample]):
    """Per-frame binary segmentation dataset over annotated YouTube-VOS frames.

    Args:
        records: VideoRecords from ``evat.data.datasets.youtube_vos.build_video_index``.
        dataset_root: root directory frame/annotation paths are relative to.
        height, width: output spatial size (mask resized with nearest-neighbor,
            image with bilinear — see ``evat.training.transforms``).
        augment: if True, apply a deterministic (seeded) horizontal flip per sample.
        seed: base seed for deterministic augmentation; the same
            ``(seed, index)`` pair always produces the same augmentation decision.
    """

    def __init__(
        self,
        records: list[VideoRecord],
        dataset_root: str | Path,
        height: int = 128,
        width: int = 128,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.frames = _annotated_frames(records)
        self.dataset_root = Path(dataset_root)
        self.height = height
        self.width = width
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> SegmentationSample:
        frame = self.frames[index]

        image_path = self.dataset_root / frame.image_path
        with Image.open(image_path) as img:
            image_array = np.array(img.convert("RGB"))  # [H, W, C]
        image = torch.from_numpy(image_array).permute(2, 0, 1).float()  # [C, H, W]

        assert frame.annotation_path is not None  # guaranteed by _annotated_frames
        mask_path = self.dataset_root / frame.annotation_path
        with Image.open(mask_path) as mask_img:
            object_id_array = np.array(mask_img)  # [H, W], palette index == object ID
        object_id_mask = torch.from_numpy(object_id_array.astype(np.int64))
        binary_mask = (object_id_mask > 0).float().unsqueeze(0)  # [1, H, W]

        image = resize_image(image, (self.height, self.width))
        binary_mask = resize_mask(binary_mask, (self.height, self.width))
        object_id_mask = (
            resize_mask(object_id_mask.unsqueeze(0).float(), (self.height, self.width))
            .squeeze(0)
            .long()
        )

        if self.augment:
            deterministic_flip = (self.seed + index) % 2 == 0
            if deterministic_flip:
                image, binary_mask = horizontal_flip(image, binary_mask)
                object_id_mask = torch.flip(object_id_mask, dims=[-1])

        image = normalize_image(image)

        return SegmentationSample(
            video_id=frame.video_id,
            frame_id=frame.frame_id,
            image=image,
            mask=binary_mask,
            object_id_mask=object_id_mask,
            object_ids=frame.object_ids,
        )
