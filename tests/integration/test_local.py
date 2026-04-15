"""
Integration tests for the local filesystem backend.

Run with:
    uv run pytest tests/integration/ -m integration -v
"""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.orc as pa_orc
import pyarrow.parquet as pq
import pytest
import torch

from torch_dataloader_utils.dataset.structured import StructuredDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(n_rows: int, row_id_offset: int = 0) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    return pa.table({
        "row_id":    pa.array(row_ids, type=pa.int32()),
        "feature_a": pa.array([float(i % 10) for i in row_ids], type=pa.float32()),
        "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
        "label":     pa.array([i % 2 for i in row_ids], type=pa.int32()),
    })


def _collect(loader) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in loader:
        for key, val in batch.items():
            if key not in result:
                result[key] = []
            if isinstance(val, torch.Tensor):
                result[key].extend(val.tolist())
            elif isinstance(val, np.ndarray):
                result[key].extend(val.tolist())
            else:
                result[key].extend(val)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def parquet_dir(tmp_path) -> tuple[Path, int]:
    """3 Parquet files × 100 rows = 300 total rows."""
    for i in range(3):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")
    return tmp_path, 300


@pytest.fixture()
def orc_dir(tmp_path) -> tuple[Path, int]:
    """2 ORC files × 100 rows = 200 total rows."""
    for i in range(2):
        pa_orc.write_table(_make_table(100, row_id_offset=i * 100), str(tmp_path / f"f{i}.orc"))
    return tmp_path, 200


@pytest.fixture()
def csv_dir(tmp_path) -> tuple[Path, int]:
    """2 CSV files × 100 rows = 200 total rows."""
    for i in range(2):
        pa_csv.write_csv(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.csv")
    return tmp_path, 200


@pytest.fixture()
def jsonl_dir(tmp_path) -> tuple[Path, int]:
    """2 JSONL files × 100 rows = 200 total rows."""
    for i in range(2):
        table = _make_table(100, row_id_offset=i * 100)
        with open(tmp_path / f"f{i}.jsonl", "w") as f:
            for row in table.to_pylist():
                f.write(json.dumps(row) + "\n")
    return tmp_path, 200


# ---------------------------------------------------------------------------
# Scenario: Full read per format
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_parquet_full_read(parquet_dir):
    path, expected = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0, batch_size=50
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == expected


@pytest.mark.integration
def test_orc_full_read(orc_dir):
    path, expected = orc_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="orc", num_workers=0, batch_size=50
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == expected


@pytest.mark.integration
def test_csv_full_read(csv_dir):
    path, expected = csv_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="csv", num_workers=0, batch_size=50
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == expected


@pytest.mark.integration
def test_jsonl_full_read(jsonl_dir):
    path, expected = jsonl_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="jsonl", num_workers=0, batch_size=50
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == expected


# ---------------------------------------------------------------------------
# Scenario: Batch is dict[str, torch.Tensor]
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_output_is_torch_tensors(parquet_dir):
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0
    )
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert isinstance(batch["row_id"], torch.Tensor)
    assert isinstance(batch["feature_a"], torch.Tensor)


# ---------------------------------------------------------------------------
# Scenario: output_format="numpy"
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_output_format_numpy(parquet_dir):
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0, output_format="numpy"
    )
    batch = next(iter(loader))
    assert isinstance(batch["row_id"], np.ndarray)


# ---------------------------------------------------------------------------
# Scenario: Column projection
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_column_projection(parquet_dir):
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0,
        columns=["feature_a", "label"],
    )
    batch = next(iter(loader))
    assert set(batch.keys()) == {"feature_a", "label"}
    assert "feature_b" not in batch
    assert "row_id" not in batch


# ---------------------------------------------------------------------------
# Scenario: Predicate pushdown
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_predicate_pushdown(parquet_dir):
    path, _ = parquet_dir
    # feature_b = row_id % 100, so feature_b >= 50 keeps rows where row_id % 100 >= 50
    # Each of 3 files has 100 rows with feature_b 0-99 → 50 rows per file → 150 total
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0,
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 150
    assert all(v >= 50 for v in rows["feature_b"])


# ---------------------------------------------------------------------------
# Scenario: Glob pattern
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_glob_pattern(tmp_path):
    # Write both parquet and csv files
    for i in range(2):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")
        pa_csv.write_csv(_make_table(10), tmp_path / f"f{i}.csv")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "*.parquet"), format="parquet", num_workers=0
    )
    rows = _collect(loader)
    # Only parquet files: 2 × 100 = 200 rows
    assert len(rows["row_id"]) == 200


# ---------------------------------------------------------------------------
# Scenario: Single file
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_single_file(tmp_path):
    p = tmp_path / "single.parquet"
    pq.write_table(_make_table(100), p)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(p), format="parquet", num_workers=0
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 100


# ---------------------------------------------------------------------------
# Scenario: No rows dropped or duplicated
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_no_rows_dropped_or_duplicated(parquet_dir):
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0, batch_size=32
    )
    rows = _collect(loader)
    row_ids = rows["row_id"]
    assert len(row_ids) == 300
    assert sorted(row_ids) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Shuffle reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_shuffle_reproducibility(parquet_dir):
    path, _ = parquet_dir

    def _run():
        loader, _ = StructuredDataset.create_dataloader(
            path=str(path), format="parquet", num_workers=0,
            shuffle=True, shuffle_seed=42, batch_size=300,
        )
        return _collect(loader)["row_id"]

    assert _run() == _run()


@pytest.mark.integration
def test_shuffle_differs_from_no_shuffle(parquet_dir):
    path, _ = parquet_dir

    def _run(shuffle):
        loader, _ = StructuredDataset.create_dataloader(
            path=str(path), format="parquet", num_workers=0,
            shuffle=shuffle, shuffle_seed=42, batch_size=300,
        )
        return _collect(loader)["row_id"]

    # Shuffled and unshuffled should differ (extremely unlikely to be identical)
    assert _run(True) != _run(False)


# ---------------------------------------------------------------------------
# Scenario: Multi-worker — no rows dropped or duplicated
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_multi_worker_no_rows_dropped(tmp_path):
    for i in range(4):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path), format="parquet", num_workers=2, batch_size=50
    )
    rows = _collect(loader)
    row_ids = rows["row_id"]
    assert len(row_ids) == 400
    assert sorted(row_ids) == list(range(400))


# ---------------------------------------------------------------------------
# Scenario: Multi-worker — more files than workers
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_multi_worker_more_files_than_workers(tmp_path):
    for i in range(6):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path), format="parquet", num_workers=2, batch_size=50
    )
    rows = _collect(loader)
    row_ids = rows["row_id"]
    assert len(row_ids) == 600
    assert sorted(row_ids) == list(range(600))


# ---------------------------------------------------------------------------
# Scenario: Imbalanced files — size-balanced splits
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_imbalanced_files_size_balanced_splits(tmp_path):
    # Files with very different row counts — [400, 300, 200, 100] = 1000 total
    row_counts = [400, 300, 200, 100]
    offset = 0
    for i, n in enumerate(row_counts):
        pq.write_table(_make_table(n, row_id_offset=offset), tmp_path / f"f{i}.parquet")
        offset += n

    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path), format="parquet", num_workers=2, batch_size=100
    )
    rows = _collect(loader)
    row_ids = rows["row_id"]

    # All 1000 rows returned with no duplicates
    assert len(row_ids) == 1000
    assert sorted(row_ids) == list(range(1000))
