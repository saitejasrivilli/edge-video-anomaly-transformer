"""Manifest generation and JSONL read/write.

Manifests are the only dataset-derived artifact this repository persists.
They contain file paths and metadata, never image/video content.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from evat.data.schemas import SampleRecord


def write_manifest(records: Iterable[SampleRecord], output_path: str | Path) -> int:
    """Write records to a JSONL manifest file. Returns the number of records written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), sort_keys=True))
            f.write("\n")
            count += 1
    return count


def read_manifest(input_path: str | Path) -> Iterator[SampleRecord]:
    """Read records from a JSONL manifest file, in file order."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed manifest line {line_number} in '{input_path}': {exc}"
                ) from exc
            yield SampleRecord(**data)
