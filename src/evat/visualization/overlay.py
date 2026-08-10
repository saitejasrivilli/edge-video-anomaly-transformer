"""Qualitative segmentation visualization.

Produces side-by-side-able PIL images (original frame, ground-truth mask,
predicted mask, colored overlay) so model failures — missed objects,
fragmented masks, boundary errors — can be inspected directly, not just
inferred from aggregate metrics. Pure numpy/PIL; no display backend
required, so this works headlessly in Colab or CI.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _denormalize_image(image: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Reverse ``evat.training.transforms.normalize_image`` for display."""
    array = ((image * std + mean).clamp(0, 1) * 255).byte().cpu().numpy()
    return np.transpose(array, (1, 2, 0))  # [C, H, W] -> [H, W, C]


def mask_to_image(mask: torch.Tensor) -> Image.Image:
    """Render a binary ``[1, H, W]`` or ``[H, W]`` mask as a grayscale PIL image."""
    array = mask.squeeze().clamp(0, 1).byte().cpu().numpy() * 255
    return Image.fromarray(array, mode="L")


def make_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Alpha-blend a binary mask, in ``color``, over a normalized image tensor."""
    base = _denormalize_image(image).astype(np.float32)
    mask_array = mask.squeeze().clamp(0, 1).cpu().numpy().astype(np.float32)

    overlay = base.copy()
    for channel, value in enumerate(color):
        overlay[..., channel] = base[..., channel] * (1 - alpha * mask_array) + value * (
            alpha * mask_array
        )

    return Image.fromarray(overlay.astype(np.uint8))


def make_qualitative_panel(
    image: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor
) -> Image.Image:
    """Side-by-side panel: original frame | ground truth | prediction | overlay."""
    frame_img = Image.fromarray(_denormalize_image(image))
    gt_img = mask_to_image(gt_mask).convert("RGB")
    pred_img = mask_to_image(pred_mask).convert("RGB")
    overlay_img = make_overlay(image, pred_mask)

    panels = [frame_img, gt_img, pred_img, overlay_img]
    width, height = panels[0].size
    panel = Image.new("RGB", (width * len(panels), height))
    for i, p in enumerate(panels):
        panel.paste(p.resize((width, height)), (i * width, 0))
    return panel
