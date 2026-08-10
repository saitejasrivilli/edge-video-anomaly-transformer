from pathlib import Path

import pytest

from evat.data.datasets.mvtec import build_manifest, discover_categories
from evat.data.validation import validate_manifest

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mvtec_mini"


def test_discover_categories_finds_fixture_category():
    assert discover_categories(FIXTURE_ROOT) == ["bottle"]


def test_build_manifest_from_fixture_matches_expected_counts():
    records = build_manifest(FIXTURE_ROOT, dataset_version="fixture-v0")

    train_good = [r for r in records if r.split == "train" and r.label == "good"]
    test_good = [r for r in records if r.split == "test" and r.label == "good"]
    test_defect = [r for r in records if r.split == "test" and r.label == "broken_large"]

    assert len(train_good) == 2
    assert len(test_good) == 1
    assert len(test_defect) == 1
    assert test_defect[0].annotation_path == "bottle/ground_truth/broken_large/000_mask.png"
    assert test_good[0].annotation_path is None


def test_build_manifest_from_fixture_passes_validation():
    records = build_manifest(FIXTURE_ROOT, dataset_version="fixture-v0")

    report = validate_manifest(records, dataset_root=FIXTURE_ROOT)

    assert report.is_valid, report.summary()


def test_build_manifest_raises_on_missing_root(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        build_manifest(tmp_path / "missing", dataset_version="v0")
