import torch

from evat.training.transforms import horizontal_flip, normalize_image, resize_image, resize_mask


def test_resize_image_bilinear_changes_shape():
    image = torch.rand(3, 8, 8)

    resized = resize_image(image, (4, 4))

    assert resized.shape == (3, 4, 4)


def test_resize_mask_nearest_preserves_exact_values():
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    resized = resize_mask(mask, (4, 4))

    # Nearest-neighbor must never invent values between 0 and 1.
    unique_values = set(resized.flatten().tolist())
    assert unique_values <= {0.0, 1.0}
    assert resized.shape == (4, 4)


def test_normalize_image_maps_uint8_range_to_roughly_unit_range():
    image = torch.tensor([[[0.0, 255.0]]])

    normalized = normalize_image(image)

    assert normalized.min() == -1.0
    assert normalized.max() == 1.0


def test_horizontal_flip_applies_identically_to_image_and_mask():
    image = torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3)
    mask = torch.tensor([[1.0, 0.0, 0.0]]).unsqueeze(0)

    flipped_image, flipped_mask = horizontal_flip(image, mask)

    assert flipped_image[0, 0].tolist() == [2.0, 1.0, 0.0]
    assert flipped_mask[0, 0].tolist() == [0.0, 0.0, 1.0]
