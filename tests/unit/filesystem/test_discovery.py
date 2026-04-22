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


# ---------------------------------------------------------------------------
# Scenario: fsspec error surfacing
# ---------------------------------------------------------------------------

def _mock_fs_with_exists(exc_on_ls=None, exists_returns=True, is_dir=True):
    """Build a mock fsspec filesystem that raises on ls/stat."""
    fs = MagicMock()
    fs.protocol = "s3"
    fs.exists.return_value = exists_returns
    fs.isdir.return_value = is_dir
    if exc_on_ls:
        fs.ls.side_effect = exc_on_ls
        fs.stat.side_effect = exc_on_ls
        fs.glob.side_effect = exc_on_ls
    return fs


def test_credential_error_raises_permission_error():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("No credentials found — AWS env vars not set"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(PermissionError, match="No credentials found"):
            discover_files("s3://bucket/data/")


def test_access_denied_403_raises_permission_error():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("403 Forbidden — access denied"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(PermissionError, match="Access denied"):
            discover_files("s3://bucket/data/")


def test_permission_denied_raises_permission_error():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("Permission denied to read bucket"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(PermissionError, match="Access denied"):
            discover_files("s3://bucket/data/")


def test_404_not_found_raises_file_not_found():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("404 NoSuchBucket"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            discover_files("s3://bucket/data/")


def test_timeout_raises_timeout_error():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("Connection timed out after 30s"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(TimeoutError, match="timed out"):
            discover_files("s3://bucket/data/")


def test_ssl_error_raises_os_error():
    fs = _mock_fs_with_exists(exc_on_ls=Exception("SSL handshake failed"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(OSError, match="SSL error"):
            discover_files("s3://bucket/data/")


def test_unknown_error_re_raised_unchanged():
    fs = _mock_fs_with_exists(exc_on_ls=RuntimeError("Something weird happened"))
    with patch("fsspec.url_to_fs", return_value=(fs, "bucket/data")):
        with pytest.raises(RuntimeError, match="Something weird happened"):
            discover_files("s3://bucket/data/")


def test_file_not_found_passthrough_not_double_wrapped(tmp_path):
    """FileNotFoundError from our own raise must not be caught and re-wrapped."""
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        discover_files(missing)

