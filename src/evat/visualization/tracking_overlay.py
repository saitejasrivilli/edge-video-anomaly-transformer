"""Lightweight tracking visualization: frame + per-track bounding box + track ID.

Intentionally minimal — draws boxes and ID labels over the raw frame so
identity switches (a box's label changing between frames) are visually
inspectable, without building a full visualization framework.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from evat.tracking.schemas import TrackedInstance

_TRACK_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def draw_tracks(frame_rgb: np.ndarray, instances: list[TrackedInstance]) -> Image.Image:
    """Draw each tracked instance's bounding box and track ID over ``frame_rgb``.

    Args:
        frame_rgb: ``[H, W, 3]`` uint8 RGB image.
        instances: this frame's ``TrackedInstance`` outputs from ``Tracker.update``.
    """
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)

    for instance in instances:
        color = _color_for_track(instance.track_id)
        if instance.bbox is not None:
            x1, y1, x2, y2 = instance.bbox
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=1)
            draw.text((x1, max(0, y1 - 10)), f"id={instance.track_id}", fill=color)

    return image
