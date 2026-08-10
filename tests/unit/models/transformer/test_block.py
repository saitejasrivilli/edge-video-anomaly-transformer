import torch

from evat.models.transformer.block import TransformerBlock


def test_block_forward_preserves_shape():
    block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
    x = torch.randn(2, 5, 16)

    out, attn = block(x)

    assert out.shape == x.shape
    assert attn is None


def test_block_gradients_are_finite():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(2, 4, 8, requires_grad=True)

    out, _ = block(x)
    out.sum().backward()

    assert torch.isfinite(x.grad).all()
    for param in block.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_block_returns_attention_when_requested():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(1, 3, 8)

    _, attn = block(x, return_attention=True)

    assert attn.shape == (1, 2, 3, 3)
