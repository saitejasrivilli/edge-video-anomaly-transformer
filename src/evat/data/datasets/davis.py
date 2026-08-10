"""DAVIS 2017 dataset adapter — DISABLED pending license verification.

docs/datasets.md records contradictory licensing claims found for DAVIS
2017 (a BSD statement scoped to the evaluation toolkit vs. an unverified
third-party claim of "CC BY 4.0, non-commercial" — which is internally
inconsistent, since standard CC BY 4.0 has no non-commercial clause). No
authoritative license text was found on the official davischallenge.org
site.

Per this project's rule against fabricating or guessing licensing
information, this adapter is intentionally left unimplemented. Do not
enable it until docs/datasets.md records a verified license from an
official source.
"""

from __future__ import annotations

from pathlib import Path

from evat.data.schemas import SampleRecord

DATASET_NAME = "davis_2017"


def build_manifest(root: Path, dataset_version: str) -> list[SampleRecord]:
    raise NotImplementedError(
        "DAVIS 2017 ingestion is disabled: dataset license is not verified. "
        "See docs/datasets.md for details before implementing this adapter."
    )
