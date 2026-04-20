"""Unit tests for TargetSizeSplitStrategy."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torch_dataloader_utils.splits.core import DataFileInfo, RowRange
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy, _parquet_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parquet(path: str, num_row_groups: int, rows_per_group: int = 100) -> None:
    """Write a Parquet file with a fixed number of row groups."""
    tables = [
        pa.table({"x": pa.array(list(range(rows_per_group)), pa.int32())})
        for _ in range(num_row_groups)
    ]
    writer = pq.ParquetWriter(path, tables[0].schema)
    for t in tables:
        writer.write_table(t)
    writer.close()


def _file_info(path: str) -> DataFileInfo:
    return DataFileInfo(path=path, file_size=os.path.getsize(path))


def _non_parquet(name: str = "data.csv") -> DataFileInfo:
    return DataFileInfo(path=f"/tmp/{name}", file_size=1024 * 1024)


# ---------------------------------------------------------------------------
# _parquet_chunks
# ---------------------------------------------------------------------------

class TestParquetChunks:
    def test_single_row_group_yields_one_chunk(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=1, rows_per_group=100)
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=10 * 1024 * 1024))
        assert len(chunks) == 1
        assert chunks[0].row_range is not None
        assert chunks[0].row_range.offset == 0
        assert chunks[0].row_range.length == 100

    def test_multiple_row_groups_small_target_yields_one_chunk_per_group(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        # Use target_bytes=1 so every row group exceeds the target → one chunk per group
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1))
        assert len(chunks) == 4
        offsets = [c.row_range.offset for c in chunks]
        assert offsets == [0, 100, 200, 300]
        lengths = [c.row_range.length for c in chunks]
        assert lengths == [100, 100, 100, 100]

    def test_multiple_row_groups_large_target_yields_one_chunk(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        # Use very large target so all row groups fit in one chunk
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1024 * 1024 * 1024))
        assert len(chunks) == 1
        assert chunks[0].row_range.offset == 0
        assert chunks[0].row_range.length == 400

    def test_chunks_cover_all_rows_no_gaps(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=5, rows_per_group=100)
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1))
        total = sum(c.row_range.length for c in chunks)
        assert total == 500

    def test_chunks_are_contiguous(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=5, rows_per_group=100)
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1))
        prev_end = 0
        for c in chunks:
            assert c.row_range.offset == prev_end
            prev_end = c.row_range.offset + c.row_range.length

    def test_bad_path_falls_back_to_single_chunk(self):
        bad_file = DataFileInfo(path="/nonexistent/file.parquet", file_size=1024)
        chunks = list(_parquet_chunks(bad_file, target_bytes=128 * 1024 * 1024))
        assert len(chunks) == 1
        assert chunks[0].row_range is None  # whole-file fallback


# ---------------------------------------------------------------------------
# TargetSizeSplitStrategy.generate — non-Parquet
# ---------------------------------------------------------------------------

class TestNonParquetFiles:
    def test_csv_files_become_single_chunks(self):
        files = [_non_parquet(f"f{i}.csv") for i in range(4)]
        strategy = TargetSizeSplitStrategy(target_bytes=128 * 1024 * 1024)
        shards = strategy.generate(files, num_workers=2)
        # 4 files → 4 splits → 2 per worker
        total_splits = sum(len(sh.splits) for sh in shards)
        assert total_splits == 4
        # All row_ranges are None (whole file)
        for shard in shards:
            for sp in shard.splits:
                assert sp.row_range is None

    def test_mixed_formats_non_parquet_whole_file(self, tmp_path):
        parquet_path = str(tmp_path / "f.parquet")
        _make_parquet(parquet_path, num_row_groups=1)
        files = [
            _file_info(parquet_path),
            _non_parquet("data.csv"),
            _non_parquet("data.orc"),
        ]
        strategy = TargetSizeSplitStrategy(target_bytes=1024 * 1024 * 1024)
        shards = strategy.generate(files, num_workers=1)
        # Parquet: 1 split (1 row group), CSV: 1 split, ORC: 1 split = 3 total
        assert sum(len(sh.splits) for sh in shards) == 3


# ---------------------------------------------------------------------------
# TargetSizeSplitStrategy.generate — assignment
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_splits_distributed_across_workers(self, tmp_path):
        # 4 row groups, target=1 byte → 4 splits → 2 workers get 2 each
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        files = [_file_info(path)]
        strategy = TargetSizeSplitStrategy(target_bytes=1)
        shards = strategy.generate(files, num_workers=2)
        assert len(shards) == 2
        assert len(shards[0].splits) == 2
        assert len(shards[1].splits) == 2

    def test_num_shards_always_equals_num_workers(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=3, rows_per_group=100)
        files = [_file_info(path)]
        for num_workers in [1, 2, 4, 8]:
            strategy = TargetSizeSplitStrategy(target_bytes=1)
            shards = strategy.generate(files, num_workers=num_workers)
            assert len(shards) == num_workers

    def test_more_workers_than_splits_some_shards_empty(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=1, rows_per_group=100)
        files = [_file_info(path)]
        strategy = TargetSizeSplitStrategy(target_bytes=1024 * 1024 * 1024)
        shards = strategy.generate(files, num_workers=4)
        # 1 file → 1 split → only shard[0] has work
        total_splits = sum(len(sh.splits) for sh in shards)
        assert total_splits == 1
        assert len(shards[0].splits) == 1


# ---------------------------------------------------------------------------
# Shuffle
# ---------------------------------------------------------------------------

class TestShuffle:
    def test_shuffle_reproducible_same_epoch(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6, rows_per_group=100)
        files = [_file_info(path)]
        s1 = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        s2 = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        shards1 = s1.generate(files, num_workers=2, epoch=0)
        shards2 = s2.generate(files, num_workers=2, epoch=0)
        offsets1 = [sp.row_range.offset for sp in shards1[0].splits]
        offsets2 = [sp.row_range.offset for sp in shards2[0].splits]
        assert offsets1 == offsets2

    def test_shuffle_differs_across_epochs(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6, rows_per_group=100)
        files = [_file_info(path)]
        strategy = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        shards0 = strategy.generate(files, num_workers=1, epoch=0)
        shards1 = strategy.generate(files, num_workers=1, epoch=1)
        offsets0 = [sp.row_range.offset for sp in shards0[0].splits]
        offsets1 = [sp.row_range.offset for sp in shards1[0].splits]
        assert offsets0 != offsets1

    def test_no_shuffle_preserves_order(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        files = [_file_info(path)]
        strategy = TargetSizeSplitStrategy(target_bytes=1, shuffle=False)
        shards = strategy.generate(files, num_workers=1)
        offsets = [sp.row_range.offset for sp in shards[0].splits]
        assert offsets == sorted(offsets)


# ---------------------------------------------------------------------------
# target_rows mode
# ---------------------------------------------------------------------------

class TestTargetRows:
    def test_target_rows_chunks_by_row_count(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6, rows_per_group=100)
        # 6 row groups × 100 rows = 600 rows; target_rows=200 → 3 chunks
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=None, target_rows=200))
        assert len(chunks) == 3
        assert all(c.row_range.length == 200 for c in chunks)

    def test_target_rows_covers_all_rows(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=5, rows_per_group=100)
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=None, target_rows=150))
        total = sum(c.row_range.length for c in chunks)
        assert total == 500

    def test_target_rows_takes_precedence_over_target_bytes(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        # target_bytes=1 would split every row group; target_rows=400 should give 1 chunk
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1, target_rows=400))
        assert len(chunks) == 1
        assert chunks[0].row_range.length == 400

    def test_target_rows_strategy_generates_row_based_chunks(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6, rows_per_group=100)
        files = [_file_info(path)]
        strategy = TargetSizeSplitStrategy(target_rows=200)
        shards = strategy.generate(files, num_workers=3)
        total_splits = sum(len(sh.splits) for sh in shards)
        assert total_splits == 3  # 600 rows / 200 target = 3 splits

    def test_target_rows_none_falls_back_to_target_bytes(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4, rows_per_group=100)
        # target_rows=None, large target_bytes → all groups in one chunk
        chunks = list(_parquet_chunks(_file_info(path), target_bytes=1024 * 1024 * 1024, target_rows=None))
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _parquet_chunks: zero row groups edge case
# ---------------------------------------------------------------------------

class TestEmptyParquet:
    def test_zero_row_groups_yields_one_whole_file_chunk(self, tmp_path):
        """A Parquet file with 0 row groups falls back to a single whole-file chunk."""
        path = str(tmp_path / "empty.parquet")
        # Write a valid but empty Parquet file (0 row groups)
        schema = pa.schema([pa.field("x", pa.int32())])
        writer = pq.ParquetWriter(path, schema)
        writer.close()

        chunks = list(_parquet_chunks(_file_info(path), target_bytes=128 * 1024 * 1024))
        assert len(chunks) == 1
        assert chunks[0].row_range is None
