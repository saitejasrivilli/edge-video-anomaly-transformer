#!/usr/bin/env python3
"""Validate and index a locally-prepared YouTube-VOS split.

This script does NOT download the dataset. Obtain YouTube-VOS through its
official channel per docs/datasets.md (agree to the Terms of Use, then use
the official Google Drive / CodaLab download links). This script only
validates an already-downloaded split's structure and builds a temporal
video index — no image content is read.

Usage:
    python scripts/prepare_youtube_vos.py --root /path/to/youtube_vos --split train

For full-split scans (thousands of videos), run this in Google Colab
against a mounted/uploaded copy of the dataset, not on a laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evat.data.datasets.youtube_vos import build_video_index, validate_video_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Path to YouTube-VOS root")
    parser.add_argument("--split", required=True, help="Split to validate, e.g. 'train'")
    args = parser.parse_args()

    split_dir = args.root / args.split
    if not split_dir.exists():
        print(
            f"Split directory '{split_dir}' does not exist.\n"
            "YouTube-VOS is not bundled with this repository and is not downloaded "
            "automatically. Agree to the official Terms of Use and download it "
            "per docs/datasets.md, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    videos = build_video_index(args.root, split=args.split)
    if not videos:
        print(f"No videos found under '{split_dir}'. Check the directory structure.")
        return 1

    report = validate_video_index(videos)
    print(report.summary())
    total_frames = sum(len(v.frames) for v in videos)
    print(f"videos: {len(videos)}, total frames indexed: {total_frames}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
