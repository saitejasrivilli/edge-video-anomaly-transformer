"""Generic classification train/eval loop, shared across all three Phase 7 models.

Using ONE shared loop (rather than one per model) is what makes the
comparison controlled: the same optimizer construction, loss, and
evaluation logic apply regardless of whether ``model`` is the non-temporal
MLP baseline, the GRU baseline, or the Transformer — only the model's
``forward(features, validity_mask)`` differs.
"""

from __future__ import annotations

import torch
from torch import nn

from evat.experiments.metrics import ClassificationMetrics, compute_classification_metrics
from evat.models.transformer.model import VideoTransformerOutput


def get_logits(model_output: torch.Tensor | VideoTransformerOutput) -> torch.Tensor:
    """Uniformly extract logits whether ``model`` returns a raw tensor or a
    ``VideoTransformerOutput`` — lets the same training loop drive any of
    the three Phase 7 models without special-casing."""
    if isinstance(model_output, VideoTransformerOutput):
        return model_output.logits
    return model_output


def train_classifier_step(
    model: nn.Module,
    features: torch.Tensor,
    validity_mask: torch.Tensor | None,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One optimization step. Returns the scalar cross-entropy loss."""
    model.train()
    optimizer.zero_grad()
    logits = get_logits(model(features, validity_mask))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]],
    num_classes: int,
) -> ClassificationMetrics:
    """Evaluate over a list of ``(features, validity_mask, labels)`` batches."""
    model.eval()
    all_predictions, all_labels = [], []

    for features, validity_mask, labels in batches:
        logits = get_logits(model(features, validity_mask))
        all_predictions.append(logits.argmax(dim=-1))
        all_labels.append(labels)

    predictions = (
        torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    )
    targets = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)

    return compute_classification_metrics(predictions, targets, num_classes)
