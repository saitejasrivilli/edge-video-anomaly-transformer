import pytest

from evat.data.paths import get_data_root


def test_get_data_root_with_explicit_override(tmp_path):
    assert get_data_root(tmp_path) == tmp_path


def test_get_data_root_reads_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAT_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path


def test_get_data_root_raises_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="does not exist"):
        get_data_root(missing)
