import pytest
import torch

from evat.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=3)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 1, 32, 32)


def test_unet_supports_multi_class_output():
    model = UNet(in_channels=3, out_channels=5, base_channels=4, depth=2)
    x = torch.randn(1, 3, 16, 16)

    y = model(x)

    assert y.shape == (1, 5, 16, 16)


def test_unet_backward_pass_produces_gradients():
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
    x = torch.randn(1, 3, 16, 16)

    y = model(x)
    y.sum().backward()

    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)


def test_unet_rejects_indivisible_input_size():
    model = UNet(in_channels=3, out_channels=1, base_channels=4, depth=3)
    x = torch.randn(1, 3, 10, 10)  # not divisible by 2**3

    with pytest.raises(ValueError, match="divisible"):
        model(x)


def test_unet_rejects_invalid_depth():
    with pytest.raises(ValueError, match="depth"):
        UNet(depth=0)
