import math

import numpy as np
import torch

from evat.models.transformer.attention import MultiHeadSelfAttention, scaled_dot_product_attention


def test_scaled_dot_product_attention_matches_hand_computed_values():
    """Tiny deterministic case where the expected result is computed directly
    from the attention formula, not merely shape-checked."""
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # [B=1, H=1, T=2, D_head=2]
    k = q.clone()
    v = q.clone()

    output, weights = scaled_dot_product_attention(q, k, v)

    scale = math.sqrt(2)
    raw_scores = np.array([[1.0, 0.0], [0.0, 1.0]]) / scale
    exp_scores = np.exp(raw_scores)
    expected_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    expected_output = expected_weights @ np.array([[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(weights[0, 0].numpy(), expected_weights, atol=1e-5)
    np.testing.assert_allclose(output[0, 0].numpy(), expected_output, atol=1e-5)


def test_attention_weights_sum_to_one_over_keys():
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 8)

    _, weights = scaled_dot_product_attention(q, k, v)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_key_mask_zeroes_out_masked_key_attention():
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    key_mask = torch.tensor([[[[True, False]]]])  # mask out key index 1

    _, weights = scaled_dot_product_attention(q, k, v, key_mask=key_mask)

    assert torch.allclose(weights[..., 1], torch.zeros_like(weights[..., 1]))
    assert torch.allclose(weights[..., 0], torch.ones_like(weights[..., 0]))


def test_fully_masked_row_does_not_produce_nan():
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 1, 4)
    v = torch.randn(1, 1, 1, 4)
    key_mask = torch.tensor([[[[False]]]])

    _, weights = scaled_dot_product_attention(q, k, v, key_mask=key_mask)

    assert not torch.isnan(weights).any()


def test_multihead_attention_output_shape():
    mha = MultiHeadSelfAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    output, weights = mha(x)

    assert output.shape == (2, 5, 16)
    assert weights is None  # not requested


def test_multihead_attention_returns_weights_with_correct_shape_when_requested():
    mha = MultiHeadSelfAttention(d_model=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    _, weights = mha(x, return_attention=True)

    assert weights.shape == (2, 4, 5, 5)  # [B, H, T, T]


def test_multihead_attention_rejects_indivisible_dims():
    try:
        MultiHeadSelfAttention(d_model=10, num_heads=3)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "divisible" in str(exc)


def test_multihead_attention_respects_validity_mask():
    mha = MultiHeadSelfAttention(d_model=8, num_heads=2)
    x = torch.randn(1, 3, 8)
    validity_mask = torch.tensor([[True, True, False]])

    _, weights = mha(x, validity_mask=validity_mask, return_attention=True)

    assert torch.allclose(weights[..., 2], torch.zeros_like(weights[..., 2]))
