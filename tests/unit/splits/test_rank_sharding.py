"""Unit tests for rank-aware sharding in TargetSizeSplitStrategy and RoundRobinSplitStrategy."""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from torch_dataloader_utils.splits.core import DataFileInfo
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet(path: str, num_row_groups: int, rows_per_group: int = 100) -> None:
    tables = [pa.table({"x": pa.array(list(range(rows_per_group)), pa.int32())})]
    writer = pq.ParquetWriter(path, tables[0].schema)
    for _ in range(num_row_groups):
        writer.write_table(tables[0])
    writer.close()


def _file(name: str = "f.csv") -> DataFileInfo:
    return DataFileInfo(path=f"/tmp/{name}", file_size=1024 * 1024)


def _parquet_file(path: str) -> DataFileInfo:
    return DataFileInfo(path=path, file_size=os.path.getsize(path))


def _all_splits(shards):
    return [sp for shard in shards for sp in shard.splits]


# ---------------------------------------------------------------------------
# TargetSizeSplitStrategy — rank distribution
# ---------------------------------------------------------------------------


class TestTargetSizeRankDistribution:
    def test_single_rank_identical_to_v1(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4)
        files = [_parquet_file(path)]
        s_v1 = TargetSizeSplitStrategy(target_bytes=1)
        s_r0 = TargetSizeSplitStrategy(target_bytes=1)
        shards_v1 = s_v1.generate(files, num_workers=2)
        shards_r0 = s_r0.generate(files, num_workers=2, num_ranks=1, rank=0)
        assert [len(s.splits) for s in shards_v1] == [len(s.splits) for s in shards_r0]

    def test_two_ranks_disjoint_union(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=8)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1)
        shards0 = s.generate(files, num_workers=1, num_ranks=2, rank=0)
        shards1 = s.generate(files, num_workers=1, num_ranks=2, rank=1)
        offsets0 = {sp.row_range.offset for sp in _all_splits(shards0)}
        offsets1 = {sp.row_range.offset for sp in _all_splits(shards1)}
        assert offsets0.isdisjoint(offsets1)
        assert len(offsets0) + len(offsets1) == 8  # all splits covered

    def test_two_ranks_equal_split_count(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=8)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1)
        n0 = sum(len(sh.splits) for sh in s.generate(files, num_workers=1, num_ranks=2, rank=0))
        n1 = sum(len(sh.splits) for sh in s.generate(files, num_workers=1, num_ranks=2, rank=1))
        assert n0 == 4
        assert n1 == 4

    def test_uneven_splits_differ_by_at_most_one(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=7)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1)
        counts = [
            sum(len(sh.splits) for sh in s.generate(files, num_workers=1, num_ranks=3, rank=r))
            for r in range(3)
        ]
        assert sum(counts) == 7
        assert max(counts) - min(counts) <= 1

    def test_more_ranks_than_splits_some_empty(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=2)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1024 * 1024 * 1024)  # 1 chunk
        counts = [
            sum(len(sh.splits) for sh in s.generate(files, num_workers=1, num_ranks=4, rank=r))
            for r in range(4)
        ]
        assert sum(counts) == 1
        assert counts[0] == 1
        assert counts[1] == counts[2] == counts[3] == 0

    def test_num_shards_always_equals_num_workers(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1)
        for num_workers in [1, 2, 4]:
            shards = s.generate(files, num_workers=num_workers, num_ranks=2, rank=0)
            assert len(shards) == num_workers


# ---------------------------------------------------------------------------
# TargetSizeSplitStrategy — shuffle determinism with ranks
# ---------------------------------------------------------------------------


class TestTargetSizeRankShuffle:
    def test_shuffle_deterministic_same_epoch(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6)
        files = [_parquet_file(path)]
        s1 = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        s2 = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        off1 = [sp.row_range.offset for sp in _all_splits(
            s1.generate(files, num_workers=1, num_ranks=2, rank=0, epoch=0)
        )]
        off2 = [sp.row_range.offset for sp in _all_splits(
            s2.generate(files, num_workers=1, num_ranks=2, rank=0, epoch=0)
        )]
        assert off1 == off2

    def test_shuffle_disjoint_across_ranks(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        off0 = {sp.row_range.offset for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=2, rank=0, epoch=0)
        )}
        off1 = {sp.row_range.offset for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=2, rank=1, epoch=0)
        )}
        assert off0.isdisjoint(off1)
        assert len(off0) + len(off1) == 6

    def test_shuffle_differs_across_epochs(self, tmp_path):
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=6)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1, shuffle=True, seed=42)
        off0 = [sp.row_range.offset for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=1, rank=0, epoch=0)
        )]
        off1 = [sp.row_range.offset for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=1, rank=0, epoch=1)
        )]
        assert off0 != off1


# ---------------------------------------------------------------------------
# TargetSizeSplitStrategy — validation
# ---------------------------------------------------------------------------


class TestTargetSizeRankValidation:
    def test_rank_out_of_range_raises(self, tmp_path):
        files = [_file()]
        s = TargetSizeSplitStrategy()
        with pytest.raises(ValueError, match="rank"):
            s.generate(files, num_workers=1, num_ranks=2, rank=2)

    def test_rank_negative_raises(self, tmp_path):
        files = [_file()]
        s = TargetSizeSplitStrategy()
        with pytest.raises(ValueError, match="rank"):
            s.generate(files, num_workers=1, num_ranks=2, rank=-1)

    def test_num_ranks_zero_raises(self, tmp_path):
        files = [_file()]
        s = TargetSizeSplitStrategy()
        with pytest.raises(ValueError, match="num_ranks"):
            s.generate(files, num_workers=1, num_ranks=0, rank=0)

    def test_num_ranks_negative_raises(self, tmp_path):
        files = [_file()]
        s = TargetSizeSplitStrategy()
        with pytest.raises(ValueError, match="num_ranks"):
            s.generate(files, num_workers=1, num_ranks=-1, rank=0)

    def test_last_rank_index_is_valid(self, tmp_path):
        """rank == num_ranks - 1 is the last valid rank — must not raise."""
        path = str(tmp_path / "f.parquet")
        _make_parquet(path, num_row_groups=4)
        files = [_parquet_file(path)]
        s = TargetSizeSplitStrategy(target_bytes=1)
        # Should not raise
        shards = s.generate(files, num_workers=1, num_ranks=4, rank=3)
        assert len(shards) == 1


# ---------------------------------------------------------------------------
# RoundRobinSplitStrategy — rank distribution
# ---------------------------------------------------------------------------


class TestRoundRobinRankDistribution:
    def test_single_rank_identical_to_v1(self):
        files = [_file(f"f{i}.csv") for i in range(6)]
        s_v1 = RoundRobinSplitStrategy()
        s_r0 = RoundRobinSplitStrategy()
        shards_v1 = s_v1.generate(files, num_workers=2)
        shards_r0 = s_r0.generate(files, num_workers=2, num_ranks=1, rank=0)
        assert [len(s.splits) for s in shards_v1] == [len(s.splits) for s in shards_r0]

    def test_two_ranks_disjoint_union(self):
        files = [_file(f"f{i}.csv") for i in range(8)]
        s = RoundRobinSplitStrategy()
        paths0 = {sp.file.path for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=2, rank=0)
        )}
        paths1 = {sp.file.path for sp in _all_splits(
            s.generate(files, num_workers=1, num_ranks=2, rank=1)
        )}
        assert paths0.isdisjoint(paths1)
        assert len(paths0) + len(paths1) == 8

    def test_more_ranks_than_files_some_empty(self):
        files = [_file(f"f{i}.csv") for i in range(2)]
        s = RoundRobinSplitStrategy()
        counts = [
            sum(len(sh.splits) for sh in s.generate(files, num_workers=1, num_ranks=4, rank=r))
            for r in range(4)
        ]
        assert sum(counts) == 2
        assert counts[2] == counts[3] == 0

    def test_rank_out_of_range_raises(self):
        files = [_file()]
        s = RoundRobinSplitStrategy()
        with pytest.raises(ValueError, match="rank"):
            s.generate(files, num_workers=1, num_ranks=2, rank=2)

    def test_rank_negative_raises(self):
        files = [_file()]
        s = RoundRobinSplitStrategy()
        with pytest.raises(ValueError, match="rank"):
            s.generate(files, num_workers=1, num_ranks=2, rank=-1)

    def test_num_ranks_zero_raises(self):
        files = [_file()]
        s = RoundRobinSplitStrategy()
        with pytest.raises(ValueError, match="num_ranks"):
            s.generate(files, num_workers=1, num_ranks=0, rank=0)
