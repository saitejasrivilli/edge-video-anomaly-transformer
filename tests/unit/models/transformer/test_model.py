import torch

from evat.models.transformer.config import TransformerConfig
from evat.models.transformer.model import VideoTransformer, VideoTransformerOutput


def _tiny_config(**overrides) -> TransformerConfig:
    defaults = dict(
        feature_dim=8,
        d_model=8,
        num_heads=2,
        num_layers=2,
        d_ff=16,
        max_sequence_length=10,
        num_classes=3,
        dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_full_model_forward_returns_named_output():
    model = VideoTransformer(_tiny_config())
    features = torch.randn(2, 5, 8)

    output = model(features)

    assert isinstance(output, VideoTransformerOutput)
    assert output.logits.shape == (2, 3)
    assert output.pooled.shape == (2, 8)
    assert output.temporal_representations.shape == (2, 5, 8)
    assert output.attention_weights is None


def test_full_model_respects_validity_mask_shape():
    model = VideoTransformer(_tiny_config())
    features = torch.randn(1, 5, 8)
    validity_mask = torch.tensor([[True, True, True, False, False]])

    output = model(features, validity_mask=validity_mask)

    assert output.logits.shape == (1, 3)


def test_full_model_backward_pass_produces_finite_gradients():
    model = VideoTransformer(_tiny_config())
    features = torch.randn(2, 5, 8)
    target = torch.randint(0, 3, (2,))

    output = model(features)
    loss = torch.nn.functional.cross_entropy(output.logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_gradient_step_reduces_loss_or_stays_finite():
    """Tiny CPU smoke test: forward -> loss -> backward -> optimizer step."""
    torch.manual_seed(0)
    model = VideoTransformer(_tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    features = torch.randn(4, 5, 8)
    target = torch.randint(0, 3, (4,))

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        output = model(features)
        loss = torch.nn.functional.cross_entropy(output.logits, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_model_is_deterministic_with_dropout_disabled():
    model = VideoTransformer(_tiny_config(dropout=0.0))
    model.eval()
    features = torch.randn(1, 4, 8)

    out1 = model(features)
    out2 = model(features)

    assert torch.equal(out1.logits, out2.logits)


def test_overfit_tiny_synthetic_classification_task():
    """Model should learn a trivial deterministic mapping — catches broken
    gradients, incorrect masking, or disconnected parameters. Tiny/fast for CI."""
    torch.manual_seed(0)
    model = VideoTransformer(_tiny_config(num_classes=2, dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Trivial rule: sequences with a large first-frame value are class 1, else class 0.
    features = torch.randn(8, 4, 8)
    features[:4, 0, 0] = 5.0
    features[4:, 0, 0] = -5.0
    target = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

    initial_loss = None
    final_loss = None
    for step in range(150):
        optimizer.zero_grad()
        output = model(features)
        loss = torch.nn.functional.cross_entropy(output.logits, target)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss
    with torch.no_grad():
        predictions = model(features).logits.argmax(dim=-1)
    assert (predictions == target).float().mean() >= 0.75


def test_num_parameters_is_positive():
    model = VideoTransformer(_tiny_config())

    assert model.num_parameters() > 0


def test_return_attention_flag_populates_attention_weights():
    model = VideoTransformer(_tiny_config())
    features = torch.randn(1, 4, 8)

    output = model(features, return_attention=True)

    assert output.attention_weights is not None
    assert len(output.attention_weights) == model.config.num_layers
