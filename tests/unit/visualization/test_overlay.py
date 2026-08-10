import torch
from PIL import Image

from evat.visualization.overlay import make_overlay, make_qualitative_panel, mask_to_image


def _synthetic_image() -> torch.Tensor:
    return torch.zeros(3, 4, 4)  # normalized "mid-gray" image


def test_mask_to_image_produces_grayscale_image():
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    img = mask_to_image(mask)

    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    assert img.size == (2, 2)


def test_make_overlay_returns_rgb_image_of_same_size():
    image = _synthetic_image()
    mask = torch.ones(1, 4, 4)

    overlay = make_overlay(image, mask)

    assert overlay.mode == "RGB"
    assert overlay.size == (4, 4)


def test_make_qualitative_panel_concatenates_four_views():
    image = _synthetic_image()
    gt_mask = torch.zeros(1, 4, 4)
    pred_mask = torch.ones(1, 4, 4)

    panel = make_qualitative_panel(image, gt_mask, pred_mask)

    assert panel.size == (4 * 4, 4)
