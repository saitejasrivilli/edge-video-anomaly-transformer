import torch

from evat.models.unet import UNet
from evat.training.checkpoint import load_checkpoint, save_checkpoint
from evat.training.config import SegmentationTrainingConfig


def test_checkpoint_save_and_load_restores_model_state(tmp_path):
    model = UNet(in_channels=3, out_channels=1, base_channels=4, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = SegmentationTrainingConfig()
    checkpoint_path = tmp_path / "ckpt.pt"

    save_checkpoint(checkpoint_path, model, optimizer, epoch=3, config=config, metrics={"iou": 0.5})

    restored_model = UNet(in_channels=3, out_channels=1, base_channels=4, depth=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)

    payload = load_checkpoint(checkpoint_path, restored_model, restored_optimizer)

    assert payload["epoch"] == 3
    assert payload["metrics"] == {"iou": 0.5}
    for p1, p2 in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(p1, p2)
