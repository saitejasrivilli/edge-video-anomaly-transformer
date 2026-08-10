"""MVTec AD dataset adapter.

License: CC BY-NC-SA 4.0, non-commercial use only. See docs/datasets.md for
the full verified license record. This module only locates, validates, and
indexes an MVTec AD archive that the user has already downloaded through
MVTec's official (registration-gated) process — it never downloads data
itself.

Expected on-disk structure (as distributed by MVTec):

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

No image resizing, normalization, or augmentation happens here — that is
out of scope for Phase 1.
"""

from __future__ import annotations

from pathlib import Path

from evat.data.schemas import SampleRecord

DATASET_NAME = "mvtec_ad"

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def discover_categories(root: Path) -> list[str]:
    """List category subdirectories present under the dataset root."""
    if not root.is_dir():
        raise ValueError(f"MVTec AD root does not exist or is not a directory: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _find_ground_truth(root: Path, category: str, defect_type: str, image_stem: str) -> str | None:
    gt_dir = root / category / "ground_truth" / defect_type
    if not gt_dir.is_dir():
        return None
    for candidate_suffix in (".png", ".jpg", ".jpeg"):
        candidate = gt_dir / f"{image_stem}_mask{candidate_suffix}"
        if candidate.exists():
            return str(candidate.relative_to(root))
    return None


def build_manifest(root: Path, dataset_version: str) -> list[SampleRecord]:
    """Scan an MVTec AD root directory and build sample records.

    Does not read image bytes or validate image content — only filesystem
    structure and naming conventions.
    """
    records: list[SampleRecord] = []

    for category in discover_categories(root):
        category_dir = root / category

        for split_dir_name, split in (("train", "train"), ("test", "test")):
            split_dir = category_dir / split_dir_name
            if not split_dir.is_dir():
                continue

            for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = label_dir.name
                for image_path in _list_images(label_dir):
                    sample_id = f"{category}/{split}/{label}/{image_path.stem}"
                    annotation_path = (
                        _find_ground_truth(root, category, label, image_path.stem)
                        if label != "good"
                        else None
                    )
                    records.append(
                        SampleRecord(
                            dataset=DATASET_NAME,
                            version=dataset_version,
                            split=split,
                            sample_id=sample_id,
                            category=category,
                            label=label,
                            image_path=str(image_path.relative_to(root)),
                            annotation_path=annotation_path,
                        )
                    )

    return records
