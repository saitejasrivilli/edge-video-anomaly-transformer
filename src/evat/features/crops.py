"""Object-crop extraction: frame + bbox/mask -> a resized crop ready for an encoder.

Crop strategy (documented, not incidental):

1. Take the object's bounding box (from ``evat.tracking.schemas.BoundingBox``,
   itself derived from the mask when not given explicitly).
2. Expand it by a configurable pixel ``padding`` on each side, clamped to
   the frame boundary (never reads outside the frame).
3. If ``mask_aware`` is enabled, zero out background pixels within the
   crop using the object's mask before resizing, so the encoder sees the
   object rather than surrounding clutter — otherwise the raw RGB crop
   (bbox region only) is used.
4. Resize to the encoder's configured input size with bilinear
   interpolation (this is image content, not a label mask, so
   nearest-neighbor is not required here — see ``evat.training.transforms``
   for the segmentation-mask-specific nearest-neighbor rule).

This is intentionally separate from ``evat.video`` (the generic temporal
frame reader) and from ``evat.training.transforms`` (segmentation-specific
preprocessing) — crop/resize/normalization here exist only to feed the
Phase 5 feature encoders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from evat.tracking.schemas import BoundingBox


def crop_object(
    frame_rgb: np.ndarray,
    bbox: BoundingBox,
    padding: int = 0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract a padded (and optionally mask-aware) object crop from a frame.

    Args:
        frame_rgb: ``[H, W, 3]`` uint8 image.
        bbox: ``(x_min, y_min, x_max, y_max)`` region to crop.
        padding: pixels to expand the box by on each side, clamped to the
            frame boundary.
        mask: if given, a ``[H, W]`` binary mask; background pixels
            (mask == 0) within the crop are zeroed out (mask-aware crop).

    Raises:
        ValueError: if the resulting crop is empty.
    """
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Empty crop region for bbox={bbox}, padding={padding}, frame shape={frame_rgb.shape}"
        )

    crop = frame_rgb[y1:y2, x1:x2].copy()

    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        crop[mask_crop == 0] = 0

    return crop


def resize_crop(crop_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a ``[H, W, 3]`` uint8 crop to ``size = (H, W)``, bilinear."""
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0)  # [1, C, H, W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).byte().numpy()  # [H, W, C]
