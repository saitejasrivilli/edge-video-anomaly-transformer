"""Reproducible on-disk feature cache.

A cache entry is identified by dataset, video/sample, frame, track, and a
hash of the extractor configuration (plus, where available, the Git
commit that produced it) — so a configuration change never silently
reuses stale features under the same key. Never committed to Git (feature
caches are large, regenerable artifacts — see .gitignore).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def hash_config(config: Any) -> str:
    """Stable short hash of a config (dataclass or dict), for cache-key/staleness checks."""
    payload = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_cache_key(
    dataset: str,
    video_id: str,
    frame_id: str,
    track_id: int | None,
    extractor_name: str,
    config_hash: str,
) -> str:
    """Deterministic cache key for one feature."""
    track_part = "global" if track_id is None else f"track{track_id}"
    raw = f"{dataset}|{video_id}|{frame_id}|{track_part}|{extractor_name}|{config_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class FeatureCache:
    """A flat directory of ``<key>.npy`` + ``<key>.json`` (metadata) pairs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.npy", self.cache_dir / f"{key}.json"

    def get(self, key: str, expected_config_hash: str | None = None) -> np.ndarray | None:
        """Return the cached feature, or None if missing or stale.

        Staleness: if ``expected_config_hash`` is given and doesn't match
        the metadata recorded at write time, the entry is treated as
        stale (not returned) rather than silently reused.
        """
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None

        if expected_config_hash is not None:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("config_hash") != expected_config_hash:
                return None

        return np.load(data_path)

    def put(
        self,
        key: str,
        feature: np.ndarray,
        config_hash: str,
        git_commit: str | None = None,
        **extra_metadata: Any,
    ) -> None:
        data_path, meta_path = self._paths(key)
        np.save(data_path, feature)
        metadata = {"config_hash": config_hash, "git_commit": git_commit, **extra_metadata}
        meta_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    def is_stale(self, key: str, expected_config_hash: str) -> bool:
        """True if an entry exists but was written under a different config."""
        _, meta_path = self._paths(key)
        if not meta_path.exists():
            return False
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return metadata.get("config_hash") != expected_config_hash
