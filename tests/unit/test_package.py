"""Foundation tests: package imports and metadata are valid."""

import evat


def test_package_imports() -> None:
    assert evat is not None


def test_package_has_version() -> None:
    assert isinstance(evat.__version__, str)
    assert evat.__version__ != ""
