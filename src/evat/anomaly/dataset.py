"""MVTec AD image/mask loading for the anomaly pipeline.

Reuses the Phase 1 ``SampleRecord`` manifest/adapter unchanged — no new
MVTec parser. Only adds image/mask decoding + resize, since Phase 1
deliberately does not read pixel content (manifest/validation only).

Preprocessing here is standalone (not reused from
``evat.training.transforms``, which is Phase 3's segmentation-specific
pipeline, or ``evat.features.crops``, which is Phase 5's tracked-object-
crop pipeline) — MVTec images are whole product photos, not video frames
or object crops, so this is its own small, documented preprocessing
contract: bilinear resize for images, nearest-neighbor resize for masks
(same rationale as elsewhere in this project: a mask's pixel values are
labels, not continuous intensities, so blending them would fabricate
values that were never annotated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from evat.data.schemas import SampleRecord

NORMAL_LABEL = "good"


def load_mvtec_image(
    dataset_root: str | Path, image_path: str, size: tuple[int, int]
) -> torch.Tensor:
    """Load + resize + normalize one MVTec image to ``[3, H, W]`` in roughly ``[-1, 1]``."""
    with Image.open(Path(dataset_root) / image_path) as img:
        array = np.array(img.convert("RGB"), dtype=np.float32)  # [H, W, 3]

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return (resized.squeeze(0) / 255.0 - 0.5) / 0.5


def load_mvtec_mask(
    dataset_root: str | Path, annotation_path: str, size: tuple[int, int]
) -> np.ndarray:
    """Load + resize one MVTec ground-truth mask to ``[H, W]`` binary, nearest-neighbor."""
    with Image.open(Path(dataset_root) / annotation_path) as mask_img:
        array = np.array(mask_img.convert("L"), dtype=np.float32)

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return (resized.squeeze(0).squeeze(0).numpy() > 0).astype(np.uint8)


def filter_records(
    records: list[SampleRecord], category: str, split: str, label: str | None = None
) -> list[SampleRecord]:
    """Select records for one category/split, optionally filtered by label."""
    return [
        r
        for r in records
        if r.category == category and r.split == split and (label is None or r.label == label)
    ]
