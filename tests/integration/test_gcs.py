"""
GCS integration tests using fsspec MemoryFileSystem (in-process mock).

Patches fsspec.url_to_fs to return a MemoryFileSystem for gs:// URIs, which
exercises the PyFileSystem(FSSpecHandler(fs)) wrapping path in reader.py —
the same code path taken by real gcsfs.

Run with:
    pytest tests/integration/test_gcs.py -m integration -v
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

BUCKET = "test-bucket"
PREFIX = "data"
BASE_URI = f"gs://{BUCKET}/{PREFIX}"


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


@pytest.fixture
def gcs_mem():
    """MemoryFileSystem populated with 3 Parquet files, patched into fsspec for gs://."""
    MemoryFileSystem.store.clear()
    mem = MemoryFileSystem()
    for i in range(3):
        mem.pipe(f"{BUCKET}/{PREFIX}/f{i}.parquet", _to_bytes(_make_table(100, row_id_offset=i * 100)))

    real_url_to_fs = fsspec.url_to_fs

    def _patched(path, **kwargs):
        if path.startswith("gs://"):
            return mem, path[len("gs://"):]
        return real_url_to_fs(path, **kwargs)

    with mock.patch("fsspec.url_to_fs", side_effect=_patched):
        yield mem

    MemoryFileSystem.store.clear()


# ---------------------------------------------------------------------------
# Scenario: GCS directory discovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_directory_discovery(gcs_mem):
    files = discover_files(f"{BASE_URI}/")
    assert len(files) == 3
    assert all(f.file_size > 0 for f in files)
    assert all(f.path.startswith("gs://") for f in files)


# ---------------------------------------------------------------------------
# Scenario: GCS glob pattern
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_glob_pattern(gcs_mem):
    # Inject two extra non-parquet files
    gcs_mem.pipe(f"{BUCKET}/{PREFIX}/notes.txt", b"ignore me")
    gcs_mem.pipe(f"{BUCKET}/{PREFIX}/data.csv", b"a,b\n1,2\n")

    files = discover_files(f"{BASE_URI}/*.parquet")
    assert len(files) == 3
    assert all(f.path.endswith(".parquet") for f in files)


# ---------------------------------------------------------------------------
# Scenario: GCS single file
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_single_file(gcs_mem):
    files = discover_files(f"{BASE_URI}/f0.parquet")
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Scenario: GCS path not found
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_path_not_found(gcs_mem):
    with pytest.raises(FileNotFoundError):
        discover_files(f"gs://{BUCKET}/does-not-exist/")


# ---------------------------------------------------------------------------
# Scenario: GCS end-to-end Parquet read
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_end_to_end_read(gcs_mem):
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
# Scenario: GCS column projection
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_column_projection(gcs_mem):
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
# Scenario: GCS predicate pushdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_predicate_pushdown(gcs_mem):
    loader, _ = StructuredDataset.create_dataloader(
        path=f"{BASE_URI}/",
        format="parquet",
        num_workers=0,
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert all(v >= 50 for v in rows["feature_b"])
    # 3 files × 50 passing rows each (feature_b = row_id % 100, rows 50-99 per file)
    assert len(rows["feature_b"]) == 150


# ---------------------------------------------------------------------------
# Scenario: GCS no rows dropped or duplicated
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_no_rows_dropped_or_duplicated(gcs_mem):
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
# Scenario: GCS missing gcsfs install hint
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gcs_missing_backend_install_hint():
    real_url_to_fs = fsspec.url_to_fs

    def _raise_import(path, **kwargs):
        if path.startswith("gs://"):
            raise ImportError("No module named 'gcsfs'")
        return real_url_to_fs(path, **kwargs)

    with mock.patch("fsspec.url_to_fs", side_effect=_raise_import):
        with pytest.raises(ImportError, match="gcsfs"):
            discover_files("gs://some-bucket/data/")
