import torch

from evat.models.transformer.config import TransformerConfig
from evat.models.transformer.encoder import TransformerEncoder


def test_encoder_projects_feature_dim_to_model_dim():
    config = TransformerConfig(feature_dim=32, d_model=16, num_heads=4, num_layers=2, d_ff=32)
    encoder = TransformerEncoder(config)
    x = torch.randn(2, 5, 32)

    out, attn = encoder(x)

    assert out.shape == (2, 5, 16)
    assert attn is None


def test_encoder_returns_per_layer_attention_when_requested():
    config = TransformerConfig(
        feature_dim=8, d_model=8, num_heads=2, num_layers=3, d_ff=16, max_sequence_length=10
    )
    encoder = TransformerEncoder(config)
    x = torch.randn(1, 4, 8)

    _, attn = encoder(x, return_attention=True)

    assert len(attn) == 3
    assert all(a.shape == (1, 2, 4, 4) for a in attn)


def test_encoder_handles_feature_dim_equal_to_d_model():
    config = TransformerConfig(feature_dim=8, d_model=8, num_heads=2, num_layers=1, d_ff=16)
    encoder = TransformerEncoder(config)

    assert isinstance(encoder.input_projection, torch.nn.Identity)
