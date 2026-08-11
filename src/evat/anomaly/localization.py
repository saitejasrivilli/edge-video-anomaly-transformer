"""Upsample a coarse anomaly map to image resolution for visualization/pixel metrics.

Nearest-neighbor upsampling (not bilinear): the anomaly map's native
resolution is the CNN backbone's downsampled feature grid — bilinear
interpolation would blend distance values and misleadingly imply a
smoother, more precise localization than the model actually produced.
Nearest-neighbor keeps the visible block structure honest (this is
anomaly LOCALIZATION at patch resolution, not pixel-precise semantic
segmentation — see docs/anomaly_task_definition.md).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def upsample_anomaly_map(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``anomaly_map``: ``[H', W']`` -> ``[H, W]`` (``size``), nearest-neighbor."""
    tensor = torch.from_numpy(anomaly_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy()
