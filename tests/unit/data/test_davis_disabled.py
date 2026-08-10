from pathlib import Path

import pytest

from evat.data.datasets.davis import build_manifest


def test_davis_adapter_is_disabled(tmp_path: Path):
    with pytest.raises(NotImplementedError, match="license is not verified"):
        build_manifest(tmp_path, dataset_version="any")
