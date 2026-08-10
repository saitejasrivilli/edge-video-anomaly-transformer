import torch

from evat.models.transformer.pooling import masked_mean_pool


def test_masked_mean_pool_matches_plain_mean_when_all_valid():
    x = torch.randn(2, 4, 8)

    pooled = masked_mean_pool(x, torch.ones(2, 4, dtype=torch.bool))
    plain = x.mean(dim=1)

    assert torch.allclose(pooled, plain)


def test_masked_mean_pool_ignores_padded_positions():
    """[A,B,C] and [A,B,C,pad,pad] must pool to the same result when padding is invalid."""
    a, b, c, pad = torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8)

    unpadded = torch.stack([a, b, c], dim=1)
    padded = torch.stack([a, b, c, pad, pad], dim=1)
    padded_mask = torch.tensor([[True, True, True, False, False]])

    pooled_unpadded = masked_mean_pool(unpadded)
    pooled_padded = masked_mean_pool(padded, padded_mask)

    assert torch.allclose(pooled_unpadded, pooled_padded, atol=1e-5)


def test_masked_mean_pool_handles_all_invalid_without_nan():
    x = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    pooled = masked_mean_pool(x, mask)

    assert torch.equal(pooled, torch.zeros(1, 8))
    assert not torch.isnan(pooled).any()
