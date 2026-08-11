import numpy as np
from PIL import Image

from evat.visualization.anomaly_overlay import heatmap_to_image, make_anomaly_panel


def test_heatmap_to_image_returns_grayscale():
    heatmap = np.array([[0.0, 1.0], [2.0, 3.0]])

    image = heatmap_to_image(heatmap)

    assert image.mode == "L"
    assert image.size == (2, 2)


def test_heatmap_to_image_handles_constant_map():
    heatmap = np.ones((4, 4))

    image = heatmap_to_image(heatmap)

    assert image.size == (4, 4)


def test_make_anomaly_panel_with_and_without_mask():
    image = np.zeros((3, 8, 8))
    amap = np.random.default_rng(0).uniform(size=(8, 8))

    panel_with_mask = make_anomaly_panel(image, amap, gt_mask=np.ones((8, 8), dtype=np.uint8))
    panel_without_mask = make_anomaly_panel(image, amap, gt_mask=None)

    assert isinstance(panel_with_mask, Image.Image)
    assert panel_with_mask.size == (8 * 3, 8)
    assert panel_without_mask.size == (8 * 3, 8)
