import torch

from evat.training.losses import BCEDiceLoss, dice_loss


def test_dice_loss_is_zero_for_perfect_prediction():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    loss = dice_loss(target, target)

    assert loss.item() < 1e-5


def test_dice_loss_is_high_for_opposite_prediction():
    pred = torch.zeros((1, 1, 2, 2))
    target = torch.ones((1, 1, 2, 2))

    loss = dice_loss(pred, target)

    assert loss.item() > 0.9


def test_bce_dice_loss_is_finite_and_positive():
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_bce_dice_loss_is_differentiable():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss = BCEDiceLoss()(logits, target)
    loss.backward()

    assert logits.grad is not None
