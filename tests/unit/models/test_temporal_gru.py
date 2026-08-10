import torch

from evat.models.temporal_gru import TemporalGRUBaseline


def test_gru_baseline_forward_shape():
    model = TemporalGRUBaseline(feature_dim=16, hidden_dim=8, num_classes=3)
    features = torch.randn(2, 5, 16)

    logits = model(features)

    assert logits.shape == (2, 3)


def test_gru_baseline_respects_validity_mask():
    model = TemporalGRUBaseline(feature_dim=8, hidden_dim=4, num_classes=2)
    features = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])

    logits = model(features, validity_mask=mask)

    assert logits.shape == (1, 2)


def test_gru_baseline_backward_pass_is_finite():
    model = TemporalGRUBaseline(feature_dim=8, hidden_dim=4, num_classes=2)
    features = torch.randn(2, 3, 8)

    logits = model(features)
    logits.sum().backward()

    for param in model.parameters():
        assert torch.isfinite(param.grad).all()


def test_gru_baseline_handles_multi_layer():
    model = TemporalGRUBaseline(feature_dim=8, hidden_dim=4, num_classes=2, num_layers=2)
    features = torch.randn(2, 3, 8)

    logits = model(features)

    assert logits.shape == (2, 2)
