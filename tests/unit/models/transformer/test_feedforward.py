import pytest
import torch

from evat.models.transformer.feedforward import FeedForward


def test_feedforward_preserves_shape():
    ffn = FeedForward(d_model=16, d_ff=32)
    x = torch.randn(2, 5, 16)

    out = ffn(x)

    assert out.shape == x.shape


def test_feedforward_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        FeedForward(d_model=8, d_ff=16, activation="not_a_real_activation")


def test_feedforward_gradients_are_finite():
    ffn = FeedForward(d_model=8, d_ff=16)
    x = torch.randn(2, 3, 8, requires_grad=True)

    ffn(x).sum().backward()

    assert torch.isfinite(x.grad).all()
