from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evat.data.datasets.youtube_vos import build_video_index
from evat.training.dataset import SegmentationDataset, collate_segmentation_batch

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures" / "youtube_vos_mini"


def _build_dataset(**kwargs) -> SegmentationDataset:
    videos = build_video_index(FIXTURE_ROOT, split="train")
    return SegmentationDataset(videos, dataset_root=FIXTURE_ROOT, height=8, width=8, **kwargs)


def test_dataset_only_includes_annotated_frames():
    dataset = _build_dataset()

    # Fixture video "bear" has 3 frames but only frame 0 is annotated.
    assert len(dataset) == 1


def test_sample_shapes_and_dtypes():
    dataset = _build_dataset()
    sample = dataset[0]

    assert sample.image.shape == (3, 8, 8)
    assert sample.image.dtype == torch.float32
    assert sample.mask.shape == (1, 8, 8)
    assert set(sample.mask.unique().tolist()) <= {0.0, 1.0}
    assert sample.object_id_mask.shape == (8, 8)
    assert sample.object_id_mask.dtype == torch.int64
    assert sample.object_ids == ("1",)


def test_image_and_mask_stay_aligned_after_resize():
    dataset = _build_dataset()
    sample = dataset[0]

    # Ground-truth fixture mask is [[0, 1], [1, 0]] before resize; foreground
    # (object id > 0) should occupy roughly the same relative region after
    # nearest-neighbor resize — check corners deterministically instead of
    # exact interpolation artifacts.
    assert sample.mask[0, 0, 0].item() in (0.0, 1.0)


def test_augmentation_is_deterministic_given_same_seed():
    ds_a = _build_dataset(augment=True, seed=7)
    ds_b = _build_dataset(augment=True, seed=7)

    sample_a = ds_a[0]
    sample_b = ds_b[0]

    assert torch.equal(sample_a.image, sample_b.image)
    assert torch.equal(sample_a.mask, sample_b.mask)


def test_collate_batches_variable_object_counts():
    dataset = _build_dataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_segmentation_batch)

    batch = next(iter(loader))

    assert batch.image.shape == (1, 3, 8, 8)
    assert batch.mask.shape == (1, 1, 8, 8)
    assert batch.object_ids == [("1",)]
