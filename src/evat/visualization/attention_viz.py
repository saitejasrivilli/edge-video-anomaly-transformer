"""Lightweight attention-weight visualization: time step vs. attention weight.

Debug/research utility only — attention weights are expensive to keep
around (see ``VideoTransformer(..., return_attention=True)``), so this is
not wired into any default training/inference path.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def attention_row_to_bars(
    attention_row: np.ndarray, width: int = 200, height: int = 60
) -> Image.Image:
    """Render one query position's attention weights over time as a bar chart.

    Args:
        attention_row: ``[T]`` attention weights (should sum to ~1), e.g.
            ``attention_weights[layer][batch_idx, head_idx, query_idx]``.
    """
    t = attention_row.shape[0]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if t == 0:
        return Image.fromarray(canvas)

    bar_width = max(1, width // t)
    max_weight = float(attention_row.max()) or 1.0

    for i in range(t):
        bar_height = int((attention_row[i] / max_weight) * (height - 1))
        x_start = i * bar_width
        x_end = min(width, x_start + bar_width)
        canvas[height - bar_height : height, x_start:x_end] = (30, 100, 200)

    return Image.fromarray(canvas)


def attention_matrix_to_heatmap(attention_matrix: np.ndarray) -> Image.Image:
    """Render a full ``[T, T]`` (query x key) attention matrix as a grayscale heatmap."""
    normalized = attention_matrix / (attention_matrix.max() or 1.0)
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
