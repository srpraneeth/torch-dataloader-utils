"""
Azure Blob Storage integration tests using fsspec MemoryFileSystem (in-process mock).

Patches fsspec.url_to_fs to return a MemoryFileSystem for az:// and abfs:// URIs,
exercising the PyFileSystem(FSSpecHandler(fs)) wrapping path in reader.py —
the same code path taken by real adlfs.

Run with:
    pytest tests/integration/test_azure.py -m integration -v
"""

from __future__ import annotations

import io
from unittest import mock

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
from fsspec.implementations.memory import MemoryFileSystem

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.filesystem.discovery import discover_files

CONTAINER = "test-container"
PREFIX = "data"
BASE_URI = f"az://{CONTAINER}/{PREFIX}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(n_rows: int, row_id_offset: int = 0) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    return pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "feature_a": pa.array([float(i % 10) for i in row_ids], type=pa.float32()),
            "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
            "label": pa.array([i % 2 for i in row_ids], type=pa.int32()),
        }
    )


def _to_bytes(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _collect(loader) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in loader:
        for key, val in batch.items():
            result.setdefault(key, [])
            if isinstance(val, torch.Tensor):
                result[key].extend(val.tolist())
            else:
                result[key].extend(val)
    return result


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_azure_patch(mem: MemoryFileSystem):
    """Return a patched fsspec.url_to_fs that intercepts az:// and abfs:// URIs."""
    real_url_to_fs = fsspec.url_to_fs

    def _patched(path, **kwargs):
        for scheme in ("abfs://", "az://"):
            if path.startswith(scheme):
                return mem, path[len(scheme):]
        return real_url_to_fs(path, **kwargs)

    return _patched


@pytest.fixture
def azure_mem():
    """MemoryFileSystem populated with 3 Parquet files, patched into fsspec for az://."""
    MemoryFileSystem.store.clear()
    mem = MemoryFileSystem()
    for i in range(3):
        mem.pipe(f"{CONTAINER}/{PREFIX}/f{i}.parquet", _to_bytes(_make_table(100, row_id_offset=i * 100)))

    with mock.patch("fsspec.url_to_fs", side_effect=_make_azure_patch(mem)):
        yield mem

    MemoryFileSystem.store.clear()


# ---------------------------------------------------------------------------
# Scenario: Azure directory discovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_directory_discovery(azure_mem):
    files = discover_files(f"{BASE_URI}/")
    assert len(files) == 3
    assert all(f.file_size > 0 for f in files)
    assert all(f.path.startswith("az://") for f in files)


# ---------------------------------------------------------------------------
# Scenario: Azure glob pattern
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_glob_pattern(azure_mem):
    azure_mem.pipe(f"{CONTAINER}/{PREFIX}/notes.txt", b"ignore me")
    azure_mem.pipe(f"{CONTAINER}/{PREFIX}/data.csv", b"a,b\n1,2\n")

    files = discover_files(f"{BASE_URI}/*.parquet")
    assert len(files) == 3
    assert all(f.path.endswith(".parquet") for f in files)


# ---------------------------------------------------------------------------
# Scenario: Azure single file
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_single_file(azure_mem):
    files = discover_files(f"{BASE_URI}/f0.parquet")
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Scenario: Azure path not found
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_path_not_found(azure_mem):
    with pytest.raises(FileNotFoundError):
        discover_files(f"az://{CONTAINER}/does-not-exist/")


# ---------------------------------------------------------------------------
# Scenario: Azure end-to-end Parquet read
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_end_to_end_read(azure_mem):
    loader, _ = StructuredDataset.create_dataloader(
        path=f"{BASE_URI}/",
        format="parquet",
        num_workers=0,
        batch_size=64,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 300
    assert isinstance(rows["row_id"][0], int)


# ---------------------------------------------------------------------------
# Scenario: Azure column projection
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_column_projection(azure_mem):
    loader, _ = StructuredDataset.create_dataloader(
        path=f"{BASE_URI}/",
        format="parquet",
        num_workers=0,
        columns=["feature_a", "label"],
    )
    batch = next(iter(loader))
    assert set(batch.keys()) == {"feature_a", "label"}
    assert "feature_b" not in batch
    assert "row_id" not in batch


# ---------------------------------------------------------------------------
# Scenario: Azure predicate pushdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_predicate_pushdown(azure_mem):
    loader, _ = StructuredDataset.create_dataloader(
        path=f"{BASE_URI}/",
        format="parquet",
        num_workers=0,
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert all(v >= 50 for v in rows["feature_b"])
    assert len(rows["feature_b"]) == 150


# ---------------------------------------------------------------------------
# Scenario: Azure no rows dropped or duplicated
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_no_rows_dropped_or_duplicated(azure_mem):
    loader, _ = StructuredDataset.create_dataloader(
        path=f"{BASE_URI}/",
        format="parquet",
        num_workers=0,
        batch_size=32,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 300
    assert sorted(rows["row_id"]) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Azure abfs:// scheme variant
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_abfs_scheme(azure_mem):
    """abfs:// scheme (ADLS Gen2) is handled identically to az://."""
    files = discover_files(f"abfs://{CONTAINER}/{PREFIX}/")
    assert len(files) == 3
    assert all(f.path.startswith("abfs://") for f in files)


# ---------------------------------------------------------------------------
# Scenario: Azure missing adlfs install hint
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_azure_missing_backend_install_hint():
    real_url_to_fs = fsspec.url_to_fs

    def _raise_import(path, **kwargs):
        if path.startswith(("az://", "abfs://")):
            raise ImportError("No module named 'adlfs'")
        return real_url_to_fs(path, **kwargs)

    with mock.patch("fsspec.url_to_fs", side_effect=_raise_import):
        with pytest.raises(ImportError, match="adlfs"):
            discover_files("az://some-container/data/")
