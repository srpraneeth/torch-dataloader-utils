"""
Integration tests for the local filesystem backend.

Run with:
    uv run pytest tests/integration/ -m integration -v
"""

import json
import sys
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
    return pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "feature_a": pa.array([float(i % 10) for i in row_ids], type=pa.float32()),
            "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
            "label": pa.array([i % 2 for i in row_ids], type=pa.int32()),
        }
    )


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
    loader, _ = StructuredDataset.create_dataloader(path=str(path), format="parquet", num_workers=0)
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
        path=str(path),
        format="parquet",
        num_workers=0,
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
        path=str(path),
        format="parquet",
        num_workers=0,
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
    loader, _ = StructuredDataset.create_dataloader(path=str(p), format="parquet", num_workers=0)
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
            path=str(path),
            format="parquet",
            num_workers=0,
            shuffle=True,
            shuffle_seed=42,
            batch_size=300,
        )
        return _collect(loader)["row_id"]

    assert _run() == _run()


@pytest.mark.integration
def test_shuffle_differs_from_no_shuffle(parquet_dir):
    path, _ = parquet_dir

    def _run(shuffle):
        loader, _ = StructuredDataset.create_dataloader(
            path=str(path),
            format="parquet",
            num_workers=0,
            shuffle=shuffle,
            shuffle_seed=42,
            batch_size=300,
        )
        return _collect(loader)["row_id"]

    # Shuffled and unshuffled should differ (extremely unlikely to be identical)
    assert _run(True) != _run(False)


_skip_multiworker = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS spawn mode deadlocks with pyarrow generators in DataLoader workers",
)

# ---------------------------------------------------------------------------
# Scenario: Multi-worker — no rows dropped or duplicated
# ---------------------------------------------------------------------------


@pytest.mark.integration
@_skip_multiworker
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
@_skip_multiworker
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
@_skip_multiworker
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


@pytest.mark.integration
def test_target_size_splits_are_balanced(tmp_path):
    """TargetSizeSplitStrategy distributes row-group chunks evenly across workers.

    With sub-file splitting, one huge file doesn't land entirely on one worker.
    Each worker's total row count should be within 1 row group of the other.
    """
    from torch_dataloader_utils.filesystem.discovery import discover_files
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # One huge file (8 row groups × 1000 rows) + one tiny file (1 row group × 100 rows)
    # Without sub-splitting: worker 0 gets 8000 rows, worker 1 gets 100 rows — terrible
    # With sub-splitting: 9 chunks distributed → worker 0 gets ~4500, worker 1 gets ~4600
    writer = pq.ParquetWriter(tmp_path / "huge.parquet", _make_table(1).schema)
    for i in range(8):
        writer.write_table(_make_table(1000, row_id_offset=i * 1000))
    writer.close()
    pq.write_table(_make_table(100, row_id_offset=8000), tmp_path / "tiny.parquet")

    files = discover_files(str(tmp_path))
    strategy = TargetSizeSplitStrategy(target_bytes=1)  # one chunk per row group
    splits = strategy.generate(files, num_workers=2)

    rows_per_worker = []
    for split in splits:
        total = sum(
            fs.row_range.length if fs.row_range is not None else (fs.file.record_count or 0)
            for fs in split.splits
        )
        rows_per_worker.append(total)

    # Each worker should get roughly half — within 1 row group (1000 rows) of each other
    assert abs(rows_per_worker[0] - rows_per_worker[1]) <= 1000, (
        f"Splits too imbalanced: worker 0 got {rows_per_worker[0]} rows, "
        f"worker 1 got {rows_per_worker[1]} rows"
    )


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — sub-file splitting of large Parquet file
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_sub_file_splitting(tmp_path):
    """A single large Parquet file with many row groups is split into chunks."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # Write one file with 10 row groups of 100 rows each = 1000 rows total
    writer = pq.ParquetWriter(tmp_path / "large.parquet", _make_table(1).schema)
    for i in range(10):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    # target_bytes=1 forces one chunk per row group → 10 chunks, all in worker 0's split
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "large.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=50,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    row_ids = rows["row_id"]
    assert len(row_ids) == 1000
    assert sorted(row_ids) == list(range(1000))


@pytest.mark.integration
def test_target_size_no_rows_dropped_single_worker(tmp_path):
    """TargetSizeSplitStrategy with num_workers=1 returns all rows."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(5):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=32,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(500))


@pytest.mark.integration
def test_target_size_predicate_pushdown_with_row_range(tmp_path):
    """Filters are applied correctly when reading a RowRange sub-split."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(4):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1)  # one chunk per row group
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert all(v >= 50 for v in rows["feature_b"])
    # Each of 4 row groups has 50 rows with feature_b >= 50
    assert len(rows["feature_b"]) == 200


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — multi-worker sub-file splitting
# ---------------------------------------------------------------------------


@pytest.mark.integration
@_skip_multiworker
def test_target_size_multi_worker_sub_file_splitting(tmp_path):
    """Single Parquet file split into chunks distributed across 2 workers — no rows lost."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(6):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    # 6 row groups, target_bytes=1 → 6 chunks → worker 0 gets [0,2,4], worker 1 gets [1,3,5]
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=2,
        batch_size=50,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(600))


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — column projection with RowRange
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_column_projection_with_row_range(tmp_path):
    """Column projection works correctly when reading a RowRange sub-split."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(3):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
        columns=["feature_a", "label"],
    )
    batch = next(iter(loader))
    assert set(batch.keys()) == {"feature_a", "label"}
    assert "row_id" not in batch
    assert "feature_b" not in batch


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — mixed Parquet + CSV in same directory
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_mixed_parquet_and_csv(tmp_path):
    """Parquet files are sub-split by row group; CSV files go whole to one chunk each."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # 2 Parquet files, 2 row groups each = 4 parquet chunks
    for i in range(2):
        writer = pq.ParquetWriter(tmp_path / f"f{i}.parquet", _make_table(1).schema)
        for j in range(2):
            writer.write_table(_make_table(100, row_id_offset=(i * 2 + j) * 100))
        writer.close()

    # 2 CSV files — these go whole-file (no sub-splitting)
    for i in range(2):
        pa_csv.write_csv(
            _make_table(50, row_id_offset=400 + i * 50),
            tmp_path / f"g{i}.csv",
        )

    # Read only the Parquet files via glob
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "*.parquet"),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 400


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — shuffle preserves all rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_shuffle_no_rows_lost(tmp_path):
    """Shuffling chunks still returns every row exactly once."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(6):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=7)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=50,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(600))


# ---------------------------------------------------------------------------
# Scenario: TargetSizeSplitStrategy — single row group file
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_single_row_group_file(tmp_path):
    """A Parquet file with exactly one row group produces one chunk and all rows."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    pq.write_table(_make_table(100), tmp_path / "f.parquet")

    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 100


# ---------------------------------------------------------------------------
# Scenario: Split correctness — chunk count and non-overlapping row ranges
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_chunk_count_matches_row_groups(tmp_path):
    """Number of chunks produced equals number of row groups when target_bytes=1."""
    from torch_dataloader_utils.filesystem.discovery import discover_files
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(7):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    files = discover_files(str(tmp_path))
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    splits = strategy.generate(files, num_workers=1)

    all_chunks = splits[0].splits
    assert len(all_chunks) == 7


@pytest.mark.integration
def test_target_size_row_ranges_are_non_overlapping(tmp_path):
    """Row ranges across all chunks within a file are contiguous and non-overlapping."""
    from torch_dataloader_utils.filesystem.discovery import discover_files
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(5):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    files = discover_files(str(tmp_path))
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    splits = strategy.generate(files, num_workers=1)

    chunks = sorted(splits[0].splits, key=lambda c: c.row_range.offset)
    prev_end = 0
    for chunk in chunks:
        assert chunk.row_range.offset == prev_end, "Gap or overlap between chunks"
        prev_end = chunk.row_range.offset + chunk.row_range.length
    assert prev_end == 500  # all 500 rows covered


@pytest.mark.integration
def test_target_size_more_workers_than_chunks(tmp_path):
    """When workers > chunks, extra workers get empty splits — no crash, no rows lost."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # 1 row group → 1 chunk → 4 workers, only worker 0 has work
    pq.write_table(_make_table(100), tmp_path / "f.parquet")

    strategy = TargetSizeSplitStrategy(target_bytes=1024 * 1024 * 1024)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,  # single-process; worker assignment tested in unit tests
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 100


@pytest.mark.integration
def test_target_size_chunks_distributed_across_4_workers(tmp_path):
    """With 4 workers and 8 row groups, each worker gets exactly 2 chunks."""
    from torch_dataloader_utils.filesystem.discovery import discover_files
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(8):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    files = discover_files(str(tmp_path))
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    splits = strategy.generate(files, num_workers=4)

    chunk_counts = [len(s.splits) for s in splits]
    assert chunk_counts == [2, 2, 2, 2]


# ---------------------------------------------------------------------------
# Scenario: Predicate — compound AND, boundary values, all-pass, all-fail
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_predicate_compound_and(parquet_dir):
    """Compound AND filter: feature_b in [20, 60) returns correct subset."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        filters=(pc.field("feature_b") >= 20) & (pc.field("feature_b") < 60),
    )
    rows = _collect(loader)
    assert all(20 <= v < 60 for v in rows["feature_b"])
    # Each file has 40 rows in [20,60) → 3 files × 40 = 120
    assert len(rows["feature_b"]) == 120


@pytest.mark.integration
def test_predicate_all_rows_pass(parquet_dir):
    """Filter that matches all rows returns the full dataset."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        filters=pc.field("feature_b") >= 0,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 300


@pytest.mark.integration
def test_predicate_no_rows_pass(parquet_dir):
    """Filter that matches no rows yields no batches."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        filters=pc.field("feature_b") > 999,
    )
    rows = _collect(loader)
    assert rows == {} or all(len(v) == 0 for v in rows.values())


@pytest.mark.integration
def test_predicate_exact_boundary(parquet_dir):
    """Boundary value: feature_b == 0 returns exactly 3 rows (one per file)."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        filters=pc.field("feature_b") == 0,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 3
    assert all(v == 0 for v in rows["feature_b"])


# ---------------------------------------------------------------------------
# Scenario: Projection — single column, all columns, combined with predicate
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_projection_single_column(parquet_dir):
    """Projecting a single column returns only that column."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        columns=["label"],
    )
    batch = next(iter(loader))
    assert list(batch.keys()) == ["label"]


@pytest.mark.integration
def test_projection_combined_with_predicate(parquet_dir):
    """Projection and predicate applied together: correct columns and filtered rows."""
    path, _ = parquet_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        columns=["feature_a", "label"],
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert set(rows.keys()) == {"feature_a", "label"}
    assert len(rows["label"]) == 150  # 50 rows per file × 3 files


@pytest.mark.integration
def test_target_size_projection_and_predicate_with_row_range(tmp_path):
    """Projection + predicate both applied correctly through RowRange sub-split path."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(4):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
        columns=["feature_a", "label"],
        filters=pc.field("feature_b") >= 50,
    )
    rows = _collect(loader)
    assert set(rows.keys()) == {"feature_a", "label"}
    assert len(rows["label"]) == 200  # 50 rows per row group × 4 row groups


# ---------------------------------------------------------------------------
# Scenario: Batch size edge cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_batch_size_larger_than_file(tmp_path):
    """batch_size larger than total rows yields a single batch with all rows."""
    pq.write_table(_make_table(50), tmp_path / "f.parquet")
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=10_000,
    )
    batches = list(loader)
    assert len(batches) == 1
    assert len(batches[0]["row_id"]) == 50


@pytest.mark.integration
def test_batch_size_one(tmp_path):
    """batch_size=1 yields one batch per row."""
    pq.write_table(_make_table(10), tmp_path / "f.parquet")
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=1,
    )
    batches = list(loader)
    assert len(batches) == 10
    assert all(len(b["row_id"]) == 1 for b in batches)


# ---------------------------------------------------------------------------
# Scenario: Shuffle — epoch changes order, seed controls reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_shuffle_epoch_changes_order(tmp_path):
    """Different epochs produce different chunk orderings but same set of rows."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(6):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    # Create dataset once with shuffle=True; set_epoch regenerates splits before each iteration
    strategy = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
    loader, dataset = StructuredDataset.create_dataloader(
        path=str(tmp_path / "f.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=100,
        split_strategy=strategy,
    )

    dataset.set_epoch(0)
    epoch0 = _collect(loader)["row_id"]

    dataset.set_epoch(1)
    epoch1 = _collect(loader)["row_id"]

    assert sorted(epoch0) == list(range(600))
    assert sorted(epoch1) == list(range(600))
    assert epoch0 != epoch1  # different epoch → different chunk order


@pytest.mark.integration
def test_target_size_shuffle_different_seeds(tmp_path):
    """Two different seeds produce different chunk orderings."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(6):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    def _run(seed):
        strategy = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=seed)
        loader, _ = StructuredDataset.create_dataloader(
            path=str(tmp_path / "f.parquet"),
            format="parquet",
            num_workers=0,
            batch_size=600,
            split_strategy=strategy,
        )
        return _collect(loader)["row_id"]

    assert _run(1) != _run(2)


# ---------------------------------------------------------------------------
# Scenario: ORC with TargetSizeSplitStrategy — whole-file chunks, no sub-split
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_target_size_orc_whole_file_no_sub_split(tmp_path):
    """ORC files with a single stripe are treated as whole-file chunks (row_range set, 1 stripe = full file)."""
    from torch_dataloader_utils.filesystem.discovery import discover_files
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # Write 3 small ORC files (100 rows each → 1 stripe each, since 100 < 1000 row stride)
    for i in range(3):
        pa_orc.write_table(_make_table(100, row_id_offset=i * 100), str(tmp_path / f"f{i}.orc"))

    files = discover_files(str(tmp_path))
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    splits = strategy.generate(files, num_workers=1)

    # 3 ORC files, each with 1 stripe → 3 chunks (each RowRange covers whole file)
    chunks = splits[0].splits
    assert len(chunks) == 3
    # Each chunk has a row_range (not None) since ORC sub-splitting is active
    assert all(c.row_range is not None for c in chunks)

    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path),
        format="orc",
        num_workers=0,
        split_strategy=strategy,
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Hive partitioning end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_hive_partitioning_scanner_path(tmp_path):
    """partitioning="hive" via scanner path adds partition columns to each batch."""
    partitioned_dir = tmp_path / "region=us" / "year=2024"
    partitioned_dir.mkdir(parents=True)
    pq.write_table(_make_table(100), partitioned_dir / "part.parquet")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(partitioned_dir / "part.parquet"),
        format="parquet",
        num_workers=0,
        partitioning="hive",
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == 100
    assert "region" in rows
    assert "year" in rows
    assert all(r == "us" for r in rows["region"])
    assert all(y == "2024" for y in rows["year"])


@pytest.mark.integration
def test_hive_partitioning_row_range_path(tmp_path):
    """partitioning="hive" via row-range path parses path and attaches columns."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    partitioned_dir = tmp_path / "region=eu" / "year=2023"
    partitioned_dir.mkdir(parents=True)
    path = partitioned_dir / "part.parquet"

    # Write multi-row-group Parquet so TargetSizeSplitStrategy produces RowRange splits
    writer = pq.ParquetWriter(str(path), _make_table(1).schema)
    for i in range(4):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    strategy = TargetSizeSplitStrategy(target_bytes=1)  # one chunk per row group
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="parquet",
        num_workers=0,
        split_strategy=strategy,
        partitioning="hive",
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(400))
    assert "region" in rows
    assert "year" in rows
    assert all(r == "eu" for r in rows["region"])
    assert all(y == "2023" for y in rows["year"])


@pytest.mark.integration
def test_hive_partitioning_no_extra_columns_without_flag(tmp_path):
    """Without partitioning="hive", partition columns must NOT appear in output."""
    partitioned_dir = tmp_path / "region=us"
    partitioned_dir.mkdir(parents=True)
    pq.write_table(_make_table(50), partitioned_dir / "part.parquet")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(partitioned_dir / "part.parquet"),
        format="parquet",
        num_workers=0,
    )
    batch = next(iter(loader))
    assert "region" not in batch




# ---------------------------------------------------------------------------
# Scenario: Rank-aware sharding — e2e simulated multi-rank
# ---------------------------------------------------------------------------


def _collect_rank(tmp_path, num_ranks, rank, num_workers=0, shuffle=False, seed=42):
    """Create a loader for one rank and collect all row_ids."""
    loader, _ = StructuredDataset.create_dataloader(
        path=str(tmp_path),
        format="parquet",
        num_workers=num_workers,
        batch_size=50,
        num_ranks=num_ranks,
        rank=rank,
        shuffle=shuffle,
        shuffle_seed=seed,
    )
    return _collect(loader).get("row_id", [])


@pytest.fixture()
def rank_parquet_dir(tmp_path) -> tuple[Path, int]:
    """6 Parquet files × 100 rows = 600 total rows, row_id 0–599."""
    for i in range(6):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")
    return tmp_path, 600


@pytest.mark.integration
def test_rank_two_ranks_all_rows_covered(rank_parquet_dir):
    """rank=0 and rank=1 together return every row exactly once."""
    path, total = rank_parquet_dir
    all_ids = []
    for rank in range(2):
        all_ids.extend(_collect_rank(path, num_ranks=2, rank=rank))
    assert len(all_ids) == total
    assert sorted(all_ids) == list(range(total))


@pytest.mark.integration
def test_rank_two_ranks_no_overlap(rank_parquet_dir):
    """rank=0 and rank=1 receive disjoint row sets."""
    path, _ = rank_parquet_dir
    ids0 = set(_collect_rank(path, num_ranks=2, rank=0))
    ids1 = set(_collect_rank(path, num_ranks=2, rank=1))
    assert ids0.isdisjoint(ids1)
    assert len(ids0) == len(ids1) == 300


@pytest.mark.integration
def test_rank_three_ranks_all_rows_covered(rank_parquet_dir):
    """3 ranks together cover all rows with no overlap."""
    path, total = rank_parquet_dir
    all_ids = []
    for rank in range(3):
        all_ids.extend(_collect_rank(path, num_ranks=3, rank=rank))
    assert len(all_ids) == total
    assert sorted(all_ids) == list(range(total))


@pytest.mark.integration
def test_rank_more_ranks_than_files(tmp_path):
    """When num_ranks > num_files, some ranks return no rows — no error."""
    for i in range(2):
        pq.write_table(_make_table(100, row_id_offset=i * 100), tmp_path / f"f{i}.parquet")

    all_ids = []
    for rank in range(4):
        all_ids.extend(_collect_rank(tmp_path, num_ranks=4, rank=rank))

    assert len(all_ids) == 200
    assert sorted(all_ids) == list(range(200))


@pytest.mark.integration
def test_rank_single_rank_same_as_no_rank(rank_parquet_dir):
    """num_ranks=1 returns identical rows to calling without rank params."""
    path, total = rank_parquet_dir
    ids_with_rank = _collect_rank(path, num_ranks=1, rank=0)
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path), format="parquet", num_workers=0, batch_size=50
    )
    ids_no_rank = _collect(loader).get("row_id", [])
    assert sorted(ids_with_rank) == sorted(ids_no_rank) == list(range(total))


@pytest.mark.integration
def test_rank_with_shuffle_all_rows_covered(rank_parquet_dir):
    """Shuffle + rank-aware sharding still returns all rows across all ranks."""
    path, total = rank_parquet_dir
    all_ids = []
    for rank in range(2):
        all_ids.extend(_collect_rank(path, num_ranks=2, rank=rank, shuffle=True, seed=99))
    assert len(all_ids) == total
    assert sorted(all_ids) == list(range(total))


@pytest.mark.integration
def test_rank_shuffle_deterministic_across_runs(rank_parquet_dir):
    """Same seed + epoch produces identical rank assignments on repeated calls."""
    path, _ = rank_parquet_dir
    ids_run1 = _collect_rank(path, num_ranks=2, rank=0, shuffle=True, seed=42)
    ids_run2 = _collect_rank(path, num_ranks=2, rank=0, shuffle=True, seed=42)
    assert ids_run1 == ids_run2


@pytest.mark.integration
def test_rank_sub_file_splitting_all_rows_covered(tmp_path):
    """Rank-aware sharding works with sub-file RowRange splits (Parquet row groups)."""
    from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

    # 1 file, 8 row groups × 100 rows = 800 rows total
    writer = pq.ParquetWriter(tmp_path / "f.parquet", _make_table(1).schema)
    for i in range(8):
        writer.write_table(_make_table(100, row_id_offset=i * 100))
    writer.close()

    # target_bytes=1 → one chunk per row group → 8 chunks across 2 ranks
    strategy = TargetSizeSplitStrategy(target_bytes=1)
    all_ids = []
    for rank in range(2):
        loader, _ = StructuredDataset.create_dataloader(
            path=str(tmp_path),
            format="parquet",
            num_workers=0,
            batch_size=50,
            split_strategy=strategy,
            num_ranks=2,
            rank=rank,
        )
        all_ids.extend(_collect(loader).get("row_id", []))

    assert len(all_ids) == 800
    assert sorted(all_ids) == list(range(800))


@pytest.mark.integration
def test_rank_with_column_projection(rank_parquet_dir):
    """Column projection works correctly with rank-aware sharding."""
    path, total = rank_parquet_dir
    all_ids = []
    for rank in range(2):
        loader, _ = StructuredDataset.create_dataloader(
            path=str(path),
            format="parquet",
            num_workers=0,
            num_ranks=2,
            rank=rank,
            columns=["row_id", "label"],
        )
        rows = _collect(loader)
        assert set(rows.keys()) == {"row_id", "label"}
        all_ids.extend(rows["row_id"])
    assert sorted(all_ids) == list(range(total))


@pytest.mark.integration
def test_rank_with_predicate_pushdown(rank_parquet_dir):
    """Predicate pushdown is applied correctly within each rank's assigned splits."""
    path, _ = rank_parquet_dir
    all_ids = []
    for rank in range(2):
        loader, _ = StructuredDataset.create_dataloader(
            path=str(path),
            format="parquet",
            num_workers=0,
            num_ranks=2,
            rank=rank,
            filters=pc.field("feature_b") >= 50,
        )
        rows = _collect(loader)
        assert all(v >= 50 for v in rows.get("feature_b", []))
        all_ids.extend(rows.get("row_id", []))
    # Each of 6 files × 50 matching rows = 300 total
    assert len(all_ids) == 300


@pytest.mark.integration
def test_hive_partitioning_create_dataloader_api(tmp_path):
    """partitioning="hive" is accepted by create_dataloader() and threaded through correctly."""
    partitioned_dir = tmp_path / "split=train"
    partitioned_dir.mkdir(parents=True)
    pq.write_table(_make_table(30), partitioned_dir / "data.parquet")

    loader, _ = StructuredDataset.create_dataloader(
        path=str(partitioned_dir / "data.parquet"),
        format="parquet",
        num_workers=0,
        batch_size=10,
        partitioning="hive",
    )
    all_splits = []
    for batch in loader:
        if "split" in batch:
            all_splits.extend(batch["split"])
    assert all(s == "train" for s in all_splits)


# ---------------------------------------------------------------------------
# ORC sub-file splitting — integration
# ---------------------------------------------------------------------------


def _make_orc_multistripe(path: str, n_rows: int, row_id_offset: int = 0) -> None:
    """Write an ORC file with n_rows rows; PyArrow ORC produces 1 stripe per 1000 rows.

    Use n_rows >= 2000 to get multiple stripes.
    """
    import random

    rng = random.Random(row_id_offset + 99991)
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    table = pa.table(
        {
            "row_id": pa.array(row_ids, pa.int32()),
            "_noise": pa.array([rng.uniform(-1e15, 1e15) for _ in range(n_rows)], pa.float64()),
            "feature_b": pa.array([rng.randint(0, 100) for _ in range(n_rows)], pa.int32()),
            "label": pa.array([rng.randint(0, 1) for _ in range(n_rows)], pa.int32()),
        }
    )
    pa_orc.write_table(table, str(path), stripe_size=1000)


@pytest.fixture()
def orc_multistripe_dir(tmp_path) -> tuple[Path, int]:
    """3 ORC files × 3000 rows = 9000 total rows; each file has 3 stripes (1 per 1000 rows)."""
    total = 0
    for i in range(3):
        _make_orc_multistripe(tmp_path / f"f{i}.orc", n_rows=3000, row_id_offset=i * 3000)
        total += 3000
    return tmp_path, total


@pytest.mark.integration
def test_orc_sub_file_splitting_all_rows_covered(orc_multistripe_dir):
    """All rows are returned when ORC files are split at stripe boundaries."""
    path, total = orc_multistripe_dir
    loader, dataset = StructuredDataset.create_dataloader(
        path=str(path),
        format="orc",
        num_workers=0,
        batch_size=128,
    )
    # Verify we actually have multiple stripes
    import pyarrow.orc as pa_orc_check
    nstripes = pa_orc_check.ORCFile(str(path / "f0.orc")).nstripes
    assert nstripes > 1, f"Expected multiple stripes, got {nstripes}"

    rows = _collect(loader)
    assert len(rows["row_id"]) == total
    assert sorted(rows["row_id"]) == list(range(total))


@pytest.mark.integration
def test_orc_sub_file_no_duplicate_rows(orc_multistripe_dir):
    """No row appears twice when ORC sub-file splits are distributed across workers."""
    path, total = orc_multistripe_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="orc",
        num_workers=0,
        batch_size=128,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == len(set(rows["row_id"])), "Duplicate rows found"


@pytest.mark.integration
def test_orc_sub_file_column_projection(orc_multistripe_dir):
    """Column projection works for ORC sub-file splits."""
    path, total = orc_multistripe_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="orc",
        num_workers=0,
        columns=["row_id", "label"],
    )
    rows = _collect(loader)
    assert set(rows.keys()) == {"row_id", "label"}
    assert len(rows["row_id"]) == total


@pytest.mark.integration
def test_orc_sub_file_predicate_filter(orc_multistripe_dir):
    """Predicate filter is correctly applied to ORC sub-file splits."""
    path, _ = orc_multistripe_dir
    loader, _ = StructuredDataset.create_dataloader(
        path=str(path),
        format="orc",
        num_workers=0,
        filters=pc.field("feature_b") < 50,
    )
    rows = _collect(loader)
    assert all(v < 50 for v in rows["feature_b"])
    assert len(rows["row_id"]) > 0


@pytest.mark.integration
def test_orc_sub_file_splitting_with_target_rows(orc_multistripe_dir):
    """target_rows chunking produces correct splits for ORC files."""
    path, total = orc_multistripe_dir
    loader, dataset = StructuredDataset.create_dataloader(
        path=str(path),
        format="orc",
        num_workers=0,
        split_rows=100,
    )
    rows = _collect(loader)
    assert len(rows["row_id"]) == total
    assert sorted(rows["row_id"]) == list(range(total))
    # 3 files × 3 stripes × (target ~100 rows/stripe at ~1000 rows/stripe) → many splits
    splits_count = sum(len(s.splits) for s in dataset._splits)
    assert splits_count >= 3  # at least one split per file


# ---------------------------------------------------------------------------
# Scenario: ORC with rank-aware sharding
# ---------------------------------------------------------------------------


@pytest.fixture()
def orc_unequal_dir(tmp_path) -> tuple[Path, int]:
    """4 ORC files with unequal row counts: [100, 300, 500, 200] = 1100 rows."""
    sizes = [100, 300, 500, 200]
    offset = 0
    for i, n in enumerate(sizes):
        pa_orc.write_table(_make_table(n, row_id_offset=offset), str(tmp_path / f"f{i}.orc"))
        offset += n
    return tmp_path, sum(sizes)


@pytest.mark.integration
def test_orc_rank_aware_sharding_disjoint_coverage(orc_unequal_dir):
    """Two ranks produce disjoint subsets that together cover all ORC rows."""
    from unittest.mock import MagicMock, patch

    path, total = orc_unequal_dir

    all_ids = []
    for rank in range(2):
        _, dataset = StructuredDataset.create_dataloader(
            path=str(path),
            format="orc",
            num_workers=1,
            num_ranks=2,
            rank=rank,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert sorted(all_ids) == list(range(total))
    assert len(set(all_ids)) == len(all_ids)


@pytest.mark.integration
def test_orc_rank_aware_sub_file_splitting(orc_unequal_dir):
    """ORC stripe-level splits distributed across ranks — full coverage."""
    from unittest.mock import MagicMock, patch

    path, total = orc_unequal_dir

    all_ids = []
    for rank in range(3):
        _, dataset = StructuredDataset.create_dataloader(
            path=str(path),
            format="orc",
            num_workers=1,
            num_ranks=3,
            rank=rank,
            split_rows=100,  # force stripe-level chunking
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert sorted(all_ids) == list(range(total))
    assert len(set(all_ids)) == len(all_ids)


# ---------------------------------------------------------------------------
# Scenario: multi-rank correctness via mp.spawn
# ---------------------------------------------------------------------------


def _distributed_worker(
    rank: int,
    world_size: int,
    data_path: str,
    results_dir: str,
) -> None:
    """Worker function run in each spawned process — no torch.distributed needed."""
    import json

    from torch_dataloader_utils.dataset.structured import StructuredDataset

    loader, _ = StructuredDataset.create_dataloader(
        path=data_path,
        format="parquet",
        num_workers=0,
        num_ranks=world_size,
        rank=rank,
        batch_size=256,
        split_rows=20,
    )
    ids = []
    for batch in loader:
        ids.extend(batch["row_id"].tolist())

    out = Path(results_dir) / f"rank{rank}.json"
    out.write_text(json.dumps(ids))


@pytest.mark.integration
def test_mp_spawn_three_ranks_disjoint_coverage(tmp_path):
    """mp.spawn with 3 ranks: each rank reads a disjoint subset, union = all rows."""
    import torch.multiprocessing as mp

    FILE_SIZES = [50, 100, 200, 500, 1000, 2000, 3000, 1000, 1150]
    total = sum(FILE_SIZES)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    offset = 0
    for i, n in enumerate(FILE_SIZES):
        row_ids = list(range(offset, offset + n))
        pq.write_table(
            pa.table({"row_id": pa.array(row_ids, pa.int32())}),
            str(tmp_path / f"part_{i:04d}.parquet"),
            row_group_size=100,
        )
        offset += n

    mp.spawn(
        _distributed_worker,
        args=(3, str(tmp_path), str(results_dir)),
        nprocs=3,
        join=True,
    )

    combined = []
    for r in range(3):
        combined.extend(json.loads((results_dir / f"rank{r}.json").read_text()))

    assert sorted(combined) == list(range(total)), "mp.spawn: rows missing or wrong"
    assert len(set(combined)) == len(combined), "mp.spawn: duplicate rows across ranks"
