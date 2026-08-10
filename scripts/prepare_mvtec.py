#!/usr/bin/env python3
"""Validate and index a locally-prepared MVTec AD dataset.

This script does NOT download the dataset. MVTec AD requires completing an
official registration form before download (see docs/datasets.md). This
script only checks that an already-downloaded dataset is structured as
expected, then generates a JSONL manifest for downstream phases.

Usage:
    python scripts/prepare_mvtec.py --root /path/to/mvtec_ad --version 2026-08-10 \\
        --output results/manifests/mvtec_ad.jsonl

For large-scale scans (the full ~5,000-image dataset), run this in Google
Colab against a mounted/uploaded copy of the dataset, not on a laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evat.data.datasets.mvtec import build_manifest
from evat.data.manifests import write_manifest
from evat.data.validation import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Path to MVTec AD root")
    parser.add_argument(
        "--version",
        required=True,
        help="Dataset version/date string to embed in the manifest (see docs/datasets.md)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output path for the JSONL manifest"
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(
            f"Dataset root '{args.root}' does not exist.\n"
            "MVTec AD is not bundled with this repository and is not downloaded "
            "automatically. Register and download it from the official source "
            "listed in docs/datasets.md, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    records = build_manifest(args.root, dataset_version=args.version)
    if not records:
        print(f"No samples found under '{args.root}'. Check the directory structure.")
        return 1

    report = validate_manifest(records, dataset_root=args.root)
    print(report.summary())
    if not report.is_valid:
        return 1

    count = write_manifest(records, args.output)
    print(f"Wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
