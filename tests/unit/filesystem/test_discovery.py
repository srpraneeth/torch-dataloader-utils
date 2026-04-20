import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from torch_dataloader_utils.filesystem.discovery import discover_files
from torch_dataloader_utils.splits.core import DataFileInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_files(directory: Path, names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        p = directory / name
        p.write_text("data")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Scenario: Directory discovery
# ---------------------------------------------------------------------------

def test_directory_returns_all_files(tmp_path):
    _write_files(tmp_path, ["a.parquet", "b.parquet", "c.parquet", "d.parquet", "e.parquet"])
    result = discover_files(str(tmp_path))
    assert len(result) == 5
    assert all(isinstance(f, DataFileInfo) for f in result)
    assert all(f.file_size is not None for f in result)


def test_directory_file_size_populated(tmp_path):
    p = tmp_path / "f.parquet"
    p.write_bytes(b"hello")
    result = discover_files(str(tmp_path))
    assert result[0].file_size == 5


# ---------------------------------------------------------------------------
# Scenario: Glob pattern
# ---------------------------------------------------------------------------

def test_glob_returns_matching_files(tmp_path):
    _write_files(tmp_path, ["a.parquet", "b.parquet", "c.parquet"])
    _write_files(tmp_path, ["x.csv", "y.csv"])
    result = discover_files(str(tmp_path / "*.parquet"))
    assert len(result) == 3
    assert all(f.path.endswith(".parquet") for f in result)


# ---------------------------------------------------------------------------
# Scenario: Single file
# ---------------------------------------------------------------------------

def test_single_file_returns_one(tmp_path):
    p = tmp_path / "f1.parquet"
    p.write_text("data")
    result = discover_files(str(p))
    assert len(result) == 1
    assert result[0].path == str(p)


# ---------------------------------------------------------------------------
# Scenario: Empty directory
# ---------------------------------------------------------------------------

def test_empty_directory_returns_empty_list(tmp_path):
    result = discover_files(str(tmp_path))
    assert result == []


# ---------------------------------------------------------------------------
# Scenario: Path does not exist
# ---------------------------------------------------------------------------

def test_nonexistent_path_raises_file_not_found(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        discover_files(missing)


# ---------------------------------------------------------------------------
# Scenario: Extension filtering
# ---------------------------------------------------------------------------

def test_extension_filter_includes_only_matching(tmp_path):
    _write_files(tmp_path, ["a.parquet", "b.parquet", "c.parquet"])
    _write_files(tmp_path, ["x.csv", "y.csv"])
    result = discover_files(str(tmp_path), extensions=[".parquet"])
    assert len(result) == 3
    assert all(f.path.endswith(".parquet") for f in result)


def test_extension_filter_none_returns_all(tmp_path):
    _write_files(tmp_path, ["a.parquet", "b.csv"])
    result = discover_files(str(tmp_path), extensions=None)
    assert len(result) == 2


def test_extension_filter_no_matches_returns_empty(tmp_path):
    _write_files(tmp_path, ["a.csv", "b.csv"])
    result = discover_files(str(tmp_path), extensions=[".parquet"])
    assert result == []


# ---------------------------------------------------------------------------
# Scenario: Missing optional backend
# ---------------------------------------------------------------------------

def test_missing_backend_raises_import_error_with_hint():
    with patch("fsspec.url_to_fs", side_effect=ImportError("No module named 's3fs'")):
        with pytest.raises(ImportError, match="pip install torch-dataloader-utils\\[s3\\]"):
            discover_files("s3://bucket/data/")


def test_missing_gcs_backend_raises_import_error_with_hint():
    with patch("fsspec.url_to_fs", side_effect=ImportError("No module named 'gcsfs'")):
        with pytest.raises(ImportError, match="pip install torch-dataloader-utils\\[gcs\\]"):
            discover_files("gs://bucket/data/")


def test_missing_azure_backend_raises_import_error_with_hint():
    with patch("fsspec.url_to_fs", side_effect=ImportError("No module named 'adlfs'")):
        with pytest.raises(ImportError, match="pip install torch-dataloader-utils\\[azure\\]"):
            discover_files("az://container/data/")


# ---------------------------------------------------------------------------
# Scenario: storage_options passed through
# ---------------------------------------------------------------------------

def test_storage_options_passed_to_fsspec(tmp_path):
    _write_files(tmp_path, ["a.parquet"])
    # Should not raise — storage_options={} is valid for local
    result = discover_files(str(tmp_path), storage_options={})
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Scenario: DataFileInfo paths are strings
# ---------------------------------------------------------------------------

def test_returned_paths_are_strings(tmp_path):
    _write_files(tmp_path, ["a.parquet", "b.parquet"])
    result = discover_files(str(tmp_path))
    assert all(isinstance(f.path, str) for f in result)


# ---------------------------------------------------------------------------
# Scenario: _install_hint — correct package suggested per scheme
# ---------------------------------------------------------------------------

def test_install_hint_s3():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[s3]" in _install_hint("s3://bucket/data/")


def test_install_hint_s3a():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[s3]" in _install_hint("s3a://bucket/data/")


def test_install_hint_gs():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[gcs]" in _install_hint("gs://bucket/data/")


def test_install_hint_gcs():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[gcs]" in _install_hint("gcs://bucket/data/")


def test_install_hint_az():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[azure]" in _install_hint("az://container/data/")


def test_install_hint_abfs():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    assert "[azure]" in _install_hint("abfs://container/data/")


def test_install_hint_local_path():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    hint = _install_hint("/local/path/data/")
    assert "pip install torch-dataloader-utils" in hint
    assert "[" not in hint  # no extra specifier for unknown/local


def test_install_hint_unknown_scheme():
    from torch_dataloader_utils.filesystem.discovery import _install_hint
    hint = _install_hint("hdfs://namenode/data/")
    assert "pip install torch-dataloader-utils" in hint


# ---------------------------------------------------------------------------
# Scenario: glob pattern matches only directories → returns empty list
# ---------------------------------------------------------------------------

def test_glob_skips_directories(tmp_path):
    """Glob that matches a directory (not a file) should not include it."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    _write_files(tmp_path, ["a.parquet"])
    # Use a glob that matches both files and the subdir
    result = discover_files(str(tmp_path / "*"))
    # subdir should be excluded; only the parquet file included
    assert all(Path(f.path).is_file() for f in result)
