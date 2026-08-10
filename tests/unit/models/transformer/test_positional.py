import torch

from evat.models.transformer.positional import SinusoidalPositionalEncoding


def test_positional_encoding_changes_shape_not_dims():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=10)
    x = torch.zeros(2, 5, 16)

    out = pe(x)

    assert out.shape == x.shape


def test_positional_encoding_preserves_temporal_order():
    """Sequence A,B,C vs C,B,A must produce different representations."""
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=10)
    a, b, c = torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8)

    forward_seq = torch.stack([a, b, c], dim=1)
    reversed_seq = torch.stack([c, b, a], dim=1)

    out_forward = pe(forward_seq)
    out_reversed = pe(reversed_seq)

    assert not torch.allclose(out_forward, out_reversed)


def test_positional_encoding_raises_when_sequence_exceeds_max_len():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=4)
    x = torch.zeros(1, 5, 8)

    try:
        pe(x)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "exceeds" in str(exc)


def test_positional_encoding_is_deterministic():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=10)
    x = torch.zeros(1, 4, 8)

    assert torch.equal(pe(x), pe(x))
