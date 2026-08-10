"""Environment-agnostic dataset path configuration.

Dataset roots must never be hard-coded (no ``/Users/...``, ``/home/...``, or
``/content/...`` literals in source). Instead, callers configure a data root
via the ``EVAT_DATA_ROOT`` environment variable, or pass an explicit path.
This lets the same code run locally or in Google Colab without modification.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT_ENV_VAR = "EVAT_DATA_ROOT"


def get_data_root(override: str | Path | None = None) -> Path:
    """Resolve the dataset root directory.

    Resolution order:
    1. Explicit ``override`` argument, if given.
    2. ``EVAT_DATA_ROOT`` environment variable.
    3. ``./data`` relative to the current working directory, as a last resort.

    Raises:
        ValueError: if the resolved root does not exist.
    """
    if override is not None:
        root = Path(override)
    else:
        env_value = os.environ.get(DEFAULT_DATA_ROOT_ENV_VAR)
        root = Path(env_value) if env_value else Path("data")

    if not root.exists():
        raise ValueError(
            f"Dataset root '{root}' does not exist. Set {DEFAULT_DATA_ROOT_ENV_VAR} "
            "to a valid directory, or pass an explicit path."
        )
    return root


def dataset_root(dataset_name: str, override: str | Path | None = None) -> Path:
    """Resolve the root directory for a specific dataset under the data root."""
    root = get_data_root(override) / dataset_name
    if not root.exists():
        raise ValueError(
            f"Expected dataset directory '{root}' does not exist. "
            f"Verify the dataset has been prepared under the '{dataset_name}' subdirectory."
        )
    return root
