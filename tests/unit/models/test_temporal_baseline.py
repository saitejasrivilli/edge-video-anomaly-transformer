import torch

from evat.models.temporal_baseline import TemporalMeanPoolBaseline


def test_baseline_forward_shape():
    model = TemporalMeanPoolBaseline(feature_dim=16, hidden_dim=8, num_classes=3)
    features = torch.randn(2, 5, 16)

    logits = model(features)

    assert logits.shape == (2, 3)


def test_baseline_respects_validity_mask():
    model = TemporalMeanPoolBaseline(feature_dim=8, hidden_dim=4, num_classes=2)
    features = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])

    logits = model(features, validity_mask=mask)

    assert logits.shape == (1, 2)


def test_baseline_backward_pass_is_finite():
    model = TemporalMeanPoolBaseline(feature_dim=8, hidden_dim=4, num_classes=2)
    features = torch.randn(2, 3, 8)

    logits = model(features)
    logits.sum().backward()

    for param in model.parameters():
        assert torch.isfinite(param.grad).all()
