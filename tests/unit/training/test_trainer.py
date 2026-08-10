"""Trainer unit tests — tiny synthetic tensors, CPU only, no real dataset.

These test the training MECHANICS (one optimizer step runs, loss
decreases-or-is-finite, checkpoint gets written) using synthetic random
data. This is not "training the model" on YouTube-VOS — no real dataset is
read here, and this is not a claim about model performance.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from evat.models.unet import UNet
from evat.training.config import SegmentationTrainingConfig
from evat.training.dataset import SegmentationBatch
from evat.training.losses import BCEDiceLoss
from evat.training.trainer import Trainer, train_step


class _SyntheticSegmentationDataset(Dataset):
    """Tiny synthetic dataset standing in for SegmentationDataset in unit tests."""

    def __init__(self, num_samples: int = 4, size: int = 8) -> None:
        self.num_samples = num_samples
        self.size = size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        image = torch.randn(3, self.size, self.size)
        mask = (torch.rand(1, self.size, self.size) > 0.5).float()
        return image, mask, f"video_{index}", f"frame_{index}", ("1",)


def _synthetic_collate(samples) -> SegmentationBatch:
    images, masks, video_ids, frame_ids, object_ids = zip(*samples, strict=True)
    return SegmentationBatch(
        video_ids=list(video_ids),
        frame_ids=list(frame_ids),
        image=torch.stack(images),
        mask=torch.stack(masks),
        object_id_mask=torch.zeros(len(samples), 8, 8, dtype=torch.long),
        object_ids=list(object_ids),
    )


def test_train_step_reduces_loss_over_several_iterations():
    model = UNet(in_channels=3, out_channels=1, base_channels=4, depth=2)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    image = torch.randn(2, 3, 8, 8)
    mask = (torch.rand(2, 1, 8, 8) > 0.5).float()
    batch = SegmentationBatch(
        video_ids=["a", "b"],
        frame_ids=["0", "0"],
        image=image,
        mask=mask,
        object_id_mask=torch.zeros(2, 8, 8, dtype=torch.long),
        object_ids=[("1",), ("1",)],
    )

    losses = [train_step(model, batch, loss_fn, optimizer) for _ in range(20)]

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert losses[-1] < losses[0]


def test_trainer_fit_runs_and_writes_checkpoint(tmp_path):
    torch.manual_seed(0)
    dataset = _SyntheticSegmentationDataset(num_samples=4, size=8)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_synthetic_collate)

    config = SegmentationTrainingConfig(
        epochs=1,
        batch_size=2,
        base_channels=4,
        depth=2,
        input_height=8,
        input_width=8,
        eval_every=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        depth=config.depth,
    )
    trainer = Trainer(model, BCEDiceLoss(), config)

    history = trainer.fit(loader, loader)

    assert "epoch_1" in history
    assert Path(config.checkpoint_dir, "epoch_1.pt").exists()
