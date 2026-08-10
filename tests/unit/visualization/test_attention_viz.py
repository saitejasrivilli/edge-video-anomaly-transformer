import numpy as np
from PIL import Image

from evat.visualization.attention_viz import attention_matrix_to_heatmap, attention_row_to_bars


def test_attention_row_to_bars_returns_image_of_requested_size():
    row = np.array([0.1, 0.5, 0.4])

    image = attention_row_to_bars(row, width=90, height=30)

    assert isinstance(image, Image.Image)
    assert image.size == (90, 30)


def test_attention_matrix_to_heatmap_returns_grayscale_image():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    image = attention_matrix_to_heatmap(matrix)

    assert image.mode == "L"
    assert image.size == (2, 2)
