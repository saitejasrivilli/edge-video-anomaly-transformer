"""Model-specific preprocessing, separate from the generic Phase 2 video pipeline.

Images are resized with bilinear interpolation. Masks are resized with
NEAREST-neighbor interpolation — bilinear/area interpolation on a mask
would blend label values at boundaries and invent fractional class
labels that never existed in the annotation, so nearest-neighbor is
required to keep mask values exact. Any augmentation (currently just
horizontal flip) is applied identically to the image and its mask so they
stay pixel-aligned.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

# Baseline normalization: maps uint8 [0, 255] to roughly [-1, 1] per channel.
# Not ImageNet statistics — this baseline does not use ImageNet-pretrained
# weights, so there is no reason to match that specific normalization.
DEFAULT_MEAN = 0.5
DEFAULT_STD = 0.5


def resize_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a ``[C, H, W]`` float image tensor to ``size = (H, W)``, bilinear."""
    resized = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a ``[1, H, W]`` (or ``[H, W]``) mask tensor to ``size``, nearest-neighbor."""
    squeeze_back = mask.dim() == 2
    m = mask.unsqueeze(0) if squeeze_back else mask
    resized = F.interpolate(m.unsqueeze(0).float(), size=size, mode="nearest").squeeze(0)
    return resized.squeeze(0) if squeeze_back else resized


def normalize_image(
    image: torch.Tensor, mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD
) -> torch.Tensor:
    """Scale a uint8-range ``[0, 255]`` float image to ``[0, 1]`` then normalize."""
    return (image / 255.0 - mean) / std


def horizontal_flip(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip an image ``[C, H, W]`` and its mask ``[*, H, W]`` along the width axis, together."""
    return torch.flip(image, dims=[-1]), torch.flip(mask, dims=[-1])
