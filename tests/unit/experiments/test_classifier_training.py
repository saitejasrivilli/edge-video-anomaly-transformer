import torch

from evat.experiments.classifier_training import (
    evaluate_classifier,
    get_logits,
    train_classifier_step,
)
from evat.models.temporal_baseline import TemporalMeanPoolBaseline
from evat.models.temporal_gru import TemporalGRUBaseline
from evat.models.transformer.config import TransformerConfig
from evat.models.transformer.model import VideoTransformer


def _models():
    feature_dim, num_classes = 8, 3
    return [
        TemporalMeanPoolBaseline(feature_dim=feature_dim, hidden_dim=4, num_classes=num_classes),
        TemporalGRUBaseline(feature_dim=feature_dim, hidden_dim=4, num_classes=num_classes),
        VideoTransformer(
            TransformerConfig(
                feature_dim=feature_dim,
                d_model=8,
                num_heads=2,
                num_layers=1,
                d_ff=16,
                max_sequence_length=10,
                num_classes=num_classes,
                dropout=0.0,
            )
        ),
    ]


def test_train_classifier_step_works_for_every_model_type():
    features = torch.randn(4, 5, 8)
    labels = torch.randint(0, 3, (4,))

    for model in _models():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = train_classifier_step(model, features, None, labels, optimizer)
        assert loss == loss  # not NaN


def test_evaluate_classifier_works_for_every_model_type():
    features = torch.randn(4, 5, 8)
    labels = torch.randint(0, 3, (4,))
    batches = [(features, None, labels)]

    for model in _models():
        metrics = evaluate_classifier(model, batches, num_classes=3)
        assert 0.0 <= metrics.accuracy <= 1.0


def test_get_logits_handles_both_output_types():
    from evat.models.transformer.model import VideoTransformerOutput

    raw = torch.randn(2, 3)
    assert torch.equal(get_logits(raw), raw)

    wrapped = VideoTransformerOutput(
        logits=raw,
        pooled=torch.zeros(2, 4),
        temporal_representations=torch.zeros(2, 5, 4),
        attention_weights=None,
    )
    assert torch.equal(get_logits(wrapped), raw)
