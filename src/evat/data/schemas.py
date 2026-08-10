"""Typed manifest record schemas.

A manifest identifies dataset samples on disk without copying dataset
content into this repository. Fields are intentionally generic so both
image-based (MVTec AD) and video-based (future) datasets can share a schema,
but no field is invented beyond what a given dataset actually supports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One dataset sample entry in a manifest.

    Attributes:
        dataset: Dataset name, e.g. "mvtec_ad".
        version: Dataset version/revision string as recorded in docs/datasets.md.
        split: Dataset split, e.g. "train" or "test".
        sample_id: Stable, unique identifier for this sample within the dataset+split.
        category: Dataset-defined category (e.g. MVTec AD object/texture class).
        label: Ground-truth label where applicable (e.g. "good" or a defect type).
        image_path: Path to the sample image/frame, relative to the dataset root.
        annotation_path: Path to the annotation/mask, if one exists for this sample.
    """

    dataset: str
    version: str
    split: str
    sample_id: str
    category: str
    label: str
    image_path: str
    annotation_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)
