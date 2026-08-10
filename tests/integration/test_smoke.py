"""Smoke test: package installs and is importable without GPU, data, or network."""

import importlib


def test_evat_module_loads_cleanly() -> None:
    module = importlib.import_module("evat")
    importlib.reload(module)
    assert hasattr(module, "__version__")
