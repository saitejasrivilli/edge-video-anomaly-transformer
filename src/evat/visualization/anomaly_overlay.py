"""Lightweight anomaly visualization: original image | anomaly heatmap | ground-truth mask.

Makes localization failures inspectable directly, not just inferred from
aggregate ROC-AUC/PR-AUC — see docs/anomaly_task_definition.md.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _denormalize(image: np.ndarray) -> np.ndarray:
    """Reverse ``evat.anomaly.dataset.load_mvtec_image``'s ``(x/255 - 0.5)/0.5`` normalization."""
    array = ((image * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return np.transpose(array, (1, 2, 0))  # [C, H, W] -> [H, W, C]


def heatmap_to_image(heatmap: np.ndarray) -> Image.Image:
    """Render a ``[H, W]`` anomaly score map as a grayscale image, normalized to its own range."""
    span = heatmap.max() - heatmap.min()
    normalized = (heatmap - heatmap.min()) / span if span > 0 else np.zeros_like(heatmap)
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L")


def make_anomaly_panel(
    image: np.ndarray, anomaly_map_upsampled: np.ndarray, gt_mask: np.ndarray | None = None
) -> Image.Image:
    """Panel: original image | anomaly heatmap | ground-truth mask (if available).

    Args:
        image: ``[3, H, W]`` normalized image tensor (as numpy).
        anomaly_map_upsampled: ``[H, W]`` anomaly scores, already upsampled to image resolution.
        gt_mask: ``[H, W]`` binary ground-truth mask, or None for normal images (no defect).
    """
    frame_img = Image.fromarray(_denormalize(image))
    heatmap_img = heatmap_to_image(anomaly_map_upsampled).convert("RGB")
    mask_img = (
        Image.fromarray((gt_mask * 255).astype(np.uint8), mode="L").convert("RGB")
        if gt_mask is not None
        else Image.new("RGB", frame_img.size, color=(0, 0, 0))
    )

    panels = [frame_img, heatmap_img, mask_img]
    width, height = panels[0].size
    panel = Image.new("RGB", (width * len(panels), height))
    for i, p in enumerate(panels):
        panel.paste(p.resize((width, height)), (i * width, 0))
    return panel
