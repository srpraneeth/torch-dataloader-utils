"""Unit tests for mid-epoch checkpoint and resume."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torch_dataloader_utils.dataset.base import BaseDataset, CheckpointMismatchError
from torch_dataloader_utils.observability import WorkerMetrics
from torch_dataloader_utils.splits.core import DataFileInfo, RowRange, Shard, Split
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy


# ---------------------------------------------------------------------------
# Minimal concrete BaseDataset for testing
# ---------------------------------------------------------------------------


class _StubDataset(BaseDataset):
    """Yields nothing — we only test state_dict/load_state_dict mechanics."""

    def __init__(self, files: list[DataFileInfo], num_workers: int = 2, shuffle: bool = False):
        self._files = files
        self._strategy = RoundRobinSplitStrategy(shuffle=shuffle, seed=42)
        self._num_workers = num_workers
        self._num_ranks = 1
        self._rank = 0
        self._output_format = "torch"
        self._shuffle_seed = 42
        self._init_splits_and_observability(epoch=0)

    def _iter_shard(self, shard, worker_id, metrics, pbar) -> Iterator[Any]:
        return iter([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_files(n: int) -> list[DataFileInfo]:
    return [DataFileInfo(path=f"s3://bucket/part-{i:04d}.parquet", file_size=1024) for i in range(n)]


def _make_shard_state(shard: Shard) -> dict:
    return {
        "splits": [
            {
                "path": fs.file.path,
                "row_offset": fs.row_range.offset if fs.row_range else None,
                "row_length": fs.row_range.length if fs.row_range else None,
            }
            for fs in shard.splits
        ]
    }


def _inject_completed_worker(ds: _StubDataset, worker_id: int) -> None:
    """Simulate a worker finishing by pushing WorkerMetrics to the local list."""
    ds._metrics_local.append(WorkerMetrics(worker_id=worker_id, rows_read=100))


# ---------------------------------------------------------------------------
# state_dict
# ---------------------------------------------------------------------------


def test_state_dict_empty_before_any_completion():
    ds = _StubDataset(_make_files(4))
    state = ds.state_dict()
    assert state["epoch"] == 0
    assert state["completed_shards"] == []
    assert state["_num_workers"] == 2


def test_state_dict_captures_completed_workers():
    ds = _StubDataset(_make_files(4), num_workers=2)
    _inject_completed_worker(ds, worker_id=0)
    state = ds.state_dict()
    assert len(state["completed_shards"]) == 1
    # shard 0's file paths should appear in the state
    shard0_paths = {s["path"] for s in state["completed_shards"][0]["splits"]}
    expected_paths = {fs.file.path for fs in ds._splits[0].splits}
    assert shard0_paths == expected_paths


def test_state_dict_drains_queue_not_just_local():
    ds = _StubDataset(_make_files(4), num_workers=2)
    # Simulate a worker process pushing via the queue.
    # Sleep briefly: multiprocessing.Queue uses a background feeder thread so
    # get_nowait() in the same process needs a moment to see the item.
    ds._metrics_queue.put(WorkerMetrics(worker_id=1, rows_read=50))
    time.sleep(0.1)
    state = ds.state_dict()
    assert len(state["completed_shards"]) == 1
    shard1_paths = {s["path"] for s in state["completed_shards"][0]["splits"]}
    expected_paths = {fs.file.path for fs in ds._splits[1].splits}
    assert shard1_paths == expected_paths


def test_state_dict_stores_row_ranges():
    """Shards with sub-file splits preserve row_offset and row_length."""
    file = DataFileInfo(path="s3://bucket/big.parquet", file_size=10000, record_count=500000)
    shard = Shard(
        id=0,
        splits=[
            Split(file=file, row_range=RowRange(offset=0, length=250000)),
            Split(file=file, row_range=RowRange(offset=250000, length=250000)),
        ],
    )
    ds = _StubDataset([], num_workers=1)
    ds._splits = [shard]
    _inject_completed_worker(ds, worker_id=0)

    state = ds.state_dict()
    splits = state["completed_shards"][0]["splits"]
    assert splits[0] == {"path": "s3://bucket/big.parquet", "row_offset": 0, "row_length": 250000}
    assert splits[1] == {
        "path": "s3://bucket/big.parquet",
        "row_offset": 250000,
        "row_length": 250000,
    }


# ---------------------------------------------------------------------------
# load_state_dict — happy path
# ---------------------------------------------------------------------------


def test_load_state_dict_restores_epoch():
    ds = _StubDataset(_make_files(4), num_workers=2)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()
    state["epoch"] = 5  # manually bump to test restoration

    ds2 = _StubDataset(_make_files(4), num_workers=2)
    ds2.load_state_dict(state)
    assert ds2._epoch == 5


def test_load_state_dict_marks_completed_workers():
    ds = _StubDataset(_make_files(4), num_workers=2)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    ds2 = _StubDataset(_make_files(4), num_workers=2)
    ds2.load_state_dict(state)
    assert 0 in ds2._completed_workers
    assert 1 not in ds2._completed_workers


def test_load_state_dict_matches_by_content_not_worker_id():
    """load_state_dict matches shards by file path content, not index."""
    files = _make_files(4)
    ds = _StubDataset(files, num_workers=2)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    # Same files, same num_workers — splits are deterministic, content matches
    ds2 = _StubDataset(files, num_workers=2)
    ds2.load_state_dict(state)
    assert len(ds2._completed_workers) == 1


def test_load_state_dict_skips_iter_for_completed_worker(tmp_path):
    """After load_state_dict, __iter__ returns immediately for completed workers."""
    f = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": pa.array(range(100))}), f)
    files = [DataFileInfo(path=str(f), file_size=f.stat().st_size, record_count=100)]

    ds = _StubDataset(files, num_workers=1)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    ds2 = _StubDataset(files, num_workers=1)
    ds2.load_state_dict(state)

    # Manually invoke __iter__ as if worker 0 (no DataLoader)
    result = list(ds2.__iter__())
    assert result == []


# ---------------------------------------------------------------------------
# load_state_dict — mismatch raises
# ---------------------------------------------------------------------------


def test_mismatch_raises_on_num_workers_change():
    files = _make_files(4)
    ds = _StubDataset(files, num_workers=4)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    # Reconstruct with different num_workers — splits change
    ds2 = _StubDataset(files, num_workers=2)
    with pytest.raises(CheckpointMismatchError, match="num_workers changed"):
        ds2.load_state_dict(state)


def test_mismatch_raises_on_file_list_change():
    files = _make_files(4)
    ds = _StubDataset(files, num_workers=2)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    # Reconstruct with different files
    different_files = _make_files(6)
    ds2 = _StubDataset(different_files, num_workers=2)
    with pytest.raises(CheckpointMismatchError, match="file list may have changed"):
        ds2.load_state_dict(state)


def test_mismatch_error_message_is_informative():
    files = _make_files(4)
    ds = _StubDataset(files, num_workers=4)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    ds2 = _StubDataset(files, num_workers=2)
    with pytest.raises(CheckpointMismatchError) as exc_info:
        ds2.load_state_dict(state)

    msg = str(exc_info.value)
    assert "num_workers changed" in msg
    assert "checkpoint=4" in msg
    assert "current=2" in msg
    assert "Reconstruct the dataset" in msg


# ---------------------------------------------------------------------------
# set_epoch clears completed workers
# ---------------------------------------------------------------------------


def test_set_epoch_clears_completed_workers():
    ds = _StubDataset(_make_files(4), num_workers=2)
    _inject_completed_worker(ds, 0)
    ds._drain_to_completed()
    assert 0 in ds._completed_workers

    ds.set_epoch(1)
    assert ds._completed_workers == set()


def test_set_epoch_does_not_affect_independent_load_state_dict():
    """load_state_dict after set_epoch still works correctly."""
    files = _make_files(4)
    ds = _StubDataset(files, num_workers=2)
    _inject_completed_worker(ds, 0)
    state = ds.state_dict()

    ds2 = _StubDataset(files, num_workers=2)
    ds2.set_epoch(0)       # called first — resets completed_workers
    ds2.load_state_dict(state)   # load_state_dict restores them
    assert 0 in ds2._completed_workers


# ---------------------------------------------------------------------------
# _drain_to_completed preserves metrics for get_metrics()
# ---------------------------------------------------------------------------


def test_drain_to_completed_preserves_metrics_for_get_metrics():
    ds = _StubDataset(_make_files(4), num_workers=2)
    ds._metrics_queue.put(WorkerMetrics(worker_id=1, rows_read=200))
    time.sleep(0.1)  # allow feeder thread to flush before get_nowait
    ds._drain_to_completed()

    # metrics should still be available via get_metrics()
    assert len(ds._metrics_local) == 1
    assert ds._metrics_local[0].worker_id == 1
    assert ds._metrics_local[0].rows_read == 200
