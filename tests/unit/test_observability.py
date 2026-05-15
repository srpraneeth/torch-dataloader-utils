"""Unit tests for observability — WorkerMetrics accumulation, get_metrics(), balance warning."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.observability import WorkerMetrics, log_split_assignment
from torch_dataloader_utils.splits.core import DataFileInfo, RowRange, Shard, Split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shard(worker_id: int, files: list[tuple[str, int, int]]) -> Shard:
    """Build a Shard from (path, file_size, record_count) tuples."""
    splits = [
        Split(file=DataFileInfo(path=p, file_size=fs, record_count=rc))
        for p, fs, rc in files
    ]
    total_bytes = sum(fs for _, fs, _ in files)
    total_rows = sum(rc for _, _, rc in files)
    return Shard(id=worker_id, splits=splits, row_count=total_rows, size_bytes=total_bytes)


def _make_parquet_bytes(n_rows: int) -> bytes:
    import io

    buf = io.BytesIO()
    pq.write_table(
        pa.table({"x": pa.array(range(n_rows), type=pa.int32())}),
        buf,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# WorkerMetrics — increments via read_split
# ---------------------------------------------------------------------------


def test_metrics_incremented_by_read_split(tmp_path):
    """read_split increments rows_read, batches_read, files_read, bytes_read."""
    f = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"x": pa.array(range(100), type=pa.int32())}),
        f,
    )
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size, record_count=100)
    shard = Shard(id=0, splits=[Split(file=info)], row_count=100, size_bytes=info.file_size)

    from torch_dataloader_utils.format.reader import read_split

    metrics = WorkerMetrics(worker_id=0)
    batches = list(read_split(shard, format="parquet", batch_size=32, metrics=metrics))

    assert metrics.rows_read == 100
    assert metrics.batches_read == len(batches)
    assert metrics.files_read == 1
    assert metrics.bytes_read == info.file_size


def test_metrics_row_range_proportional(tmp_path):
    """bytes_read is proportional when a row_range split is used."""
    f = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"x": pa.array(range(200), type=pa.int32())}),
        f,
        row_group_size=100,
    )
    file_size = f.stat().st_size
    info = DataFileInfo(path=str(f), file_size=file_size, record_count=200)
    row_range = RowRange(offset=0, length=100)
    shard = Shard(id=0, splits=[Split(file=info, row_range=row_range)])

    from torch_dataloader_utils.format.reader import read_split

    metrics = WorkerMetrics(worker_id=0)
    list(read_split(shard, format="parquet", batch_size=32, metrics=metrics))

    expected = int(file_size * 100 / 200)
    assert metrics.bytes_read == expected


def test_metrics_none_leaves_behaviour_unchanged(tmp_path):
    """read_split with metrics=None yields all batches without error."""
    f = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": pa.array(range(50), type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size)
    shard = Shard(id=0, splits=[Split(file=info)])

    from torch_dataloader_utils.format.reader import read_split

    batches = list(read_split(shard, format="parquet", batch_size=16, metrics=None))
    assert sum(b.num_rows for b in batches) == 50


# ---------------------------------------------------------------------------
# get_metrics() / reset_metrics()
# ---------------------------------------------------------------------------


def test_get_metrics_empty_before_iteration(tmp_path):
    """get_metrics() returns [] before any epoch has run."""
    f = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array([1, 2, 3], type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size)
    ds = StructuredDataset(files=[info], format="parquet", num_workers=0)

    assert ds.get_metrics() == []


def test_get_metrics_after_iteration(tmp_path):
    """get_metrics() returns one WorkerMetrics with correct row count after num_workers=0."""
    f = tmp_path / "f.parquet"
    n_rows = 150
    pq.write_table(pa.table({"x": pa.array(range(n_rows), type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size, record_count=n_rows)
    ds = StructuredDataset(files=[info], format="parquet", num_workers=0, batch_size=32)

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=None, num_workers=0)
    rows = sum(b["x"].shape[0] for b in loader)
    assert rows == n_rows

    metrics = ds.get_metrics()
    assert len(metrics) == 1
    m = metrics[0]
    assert m.rows_read == n_rows
    assert m.files_read == 1
    assert m.bytes_read > 0
    assert m.elapsed_sec > 0


def test_get_metrics_empty_on_second_drain(tmp_path):
    """get_metrics() returns [] when called a second time (queue already drained)."""
    f = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array(range(20), type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size)
    ds = StructuredDataset(files=[info], format="parquet", num_workers=0)

    from torch.utils.data import DataLoader

    list(DataLoader(ds, batch_size=None, num_workers=0))

    assert len(ds.get_metrics()) == 1
    assert ds.get_metrics() == []


def test_set_epoch_clears_metrics(tmp_path):
    """set_epoch() discards stale metrics from the previous epoch."""
    f = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array(range(20), type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size)
    ds = StructuredDataset(files=[info], format="parquet", num_workers=0)

    from torch.utils.data import DataLoader

    list(DataLoader(ds, batch_size=None, num_workers=0))
    # Queue has one item from epoch 0; set_epoch should clear it
    ds.set_epoch(1)
    assert ds.get_metrics() == []


# ---------------------------------------------------------------------------
# Load balance warning
# ---------------------------------------------------------------------------


def test_balance_warning_fires_on_imbalance(caplog):
    """WARNING is emitted when max/min shard bytes ratio exceeds 2.0."""
    shards = [
        _make_shard(0, [("a.parquet", 300_000_000, 1_000_000)]),  # 300 MB
        _make_shard(1, [("b.parquet", 100_000_000, 300_000)]),    # 100 MB — 3× smaller
    ]

    with caplog.at_level(logging.WARNING, logger="torch_dataloader_utils"):
        log_split_assignment(shards, epoch=0, rank=0, num_ranks=1)

    assert any("Unbalanced" in r.message for r in caplog.records)
    assert any("3.0×" in r.message for r in caplog.records)


def test_balance_warning_absent_when_balanced(caplog):
    """No WARNING when shards are within 2× of each other."""
    shards = [
        _make_shard(0, [("a.parquet", 200_000_000, 1_000_000)]),
        _make_shard(1, [("b.parquet", 180_000_000, 900_000)]),
    ]

    with caplog.at_level(logging.WARNING, logger="torch_dataloader_utils"):
        log_split_assignment(shards, epoch=0, rank=0, num_ranks=1)

    assert not any("Unbalanced" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# show_progress — ImportError when tqdm missing
# ---------------------------------------------------------------------------


def test_show_progress_raises_without_tqdm(tmp_path):
    """ImportError with install hint when show_progress=True and tqdm not installed."""
    f = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array([1], type=pa.int32())}), f)
    info = DataFileInfo(path=str(f), file_size=f.stat().st_size)

    with patch("builtins.__import__", side_effect=_block_tqdm):
        with pytest.raises(ImportError, match="tqdm"):
            StructuredDataset(
                files=[info], format="parquet", num_workers=0, show_progress=True
            )


def _block_tqdm(name, *args, **kwargs):
    if name == "tqdm":
        raise ImportError("No module named 'tqdm'")
    return __import__(name, *args, **kwargs)
