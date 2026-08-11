from pathlib import Path

from evat.anomaly.dataset import filter_records, load_mvtec_image, load_mvtec_mask
from evat.data.datasets.mvtec import build_manifest

FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures" / "mvtec_anomaly_mini"


def test_filter_records_selects_only_matching_category_split_label():
    records = build_manifest(FIXTURE_ROOT, dataset_version="fixture-v0")

    train_good_bottle = filter_records(records, category="bottle", split="train", label="good")

    assert len(train_good_bottle) == 4
    assert all(
        r.category == "bottle" and r.split == "train" and r.label == "good"
        for r in train_good_bottle
    )


def test_filter_records_isolates_categories():
    records = build_manifest(FIXTURE_ROOT, dataset_version="fixture-v0")

    bottle_records = filter_records(records, category="bottle", split="train")
    cable_records = filter_records(records, category="cable", split="train")

    bottle_paths = {r.image_path for r in bottle_records}
    cable_paths = {r.image_path for r in cable_records}
    assert bottle_paths.isdisjoint(cable_paths)


def test_load_mvtec_image_shape_and_range():
    tensor = load_mvtec_image(FIXTURE_ROOT, "bottle/train/good/000.png", size=(16, 16))

    assert tensor.shape == (3, 16, 16)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_load_mvtec_mask_is_binary_and_resized():
    mask = load_mvtec_mask(
        FIXTURE_ROOT, "bottle/ground_truth/broken_large/000_mask.png", size=(8, 8)
    )

    assert mask.shape == (8, 8)
    assert set(mask.flatten().tolist()) <= {0, 1}
    assert mask.sum() > 0  # fixture mask has a real defect region
