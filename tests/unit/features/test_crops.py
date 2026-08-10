import numpy as np
import pytest

from evat.features.crops import crop_object, resize_crop


def test_crop_object_extracts_bbox_region():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[2:5, 2:5] = 255

    crop = crop_object(frame, bbox=(2, 2, 5, 5))

    assert crop.shape == (3, 3, 3)
    assert (crop == 255).all()


def test_crop_object_applies_padding_clamped_to_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    crop = crop_object(frame, bbox=(0, 0, 2, 2), padding=5)

    assert crop.shape[0] <= 7 and crop.shape[1] <= 7  # clamped, not out of bounds


def test_crop_object_mask_aware_zeroes_background():
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    crop = crop_object(frame, bbox=(0, 0, 4, 4), mask=mask)

    assert (crop[1:3, 1:3] == 200).all()
    assert (crop[0, 0] == 0).all()


def test_crop_object_raises_on_empty_region():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Empty crop"):
        crop_object(frame, bbox=(5, 5, 5, 5))


def test_resize_crop_changes_spatial_size():
    crop = np.zeros((4, 4, 3), dtype=np.uint8)

    resized = resize_crop(crop, size=(16, 16))

    assert resized.shape == (16, 16, 3)
    assert resized.dtype == np.uint8
