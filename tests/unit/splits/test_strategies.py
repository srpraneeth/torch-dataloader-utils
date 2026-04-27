from torch_dataloader_utils.splits.balanced import SizeBalancedSplitStrategy
from torch_dataloader_utils.splits.core import DataFileInfo, IcebergDataFileInfo
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy

# --- helpers ---


def make_files(n: int) -> list[DataFileInfo]:
    return [DataFileInfo(path=f"f{i}.parquet") for i in range(n)]


def make_sized_files(sizes: list[int]) -> list[DataFileInfo]:
    return [DataFileInfo(path=f"f{i}.parquet", file_size=s) for i, s in enumerate(sizes)]


def make_counted_files(counts: list[int]) -> list[DataFileInfo]:
    return [DataFileInfo(path=f"f{i}.parquet", record_count=c) for i, c in enumerate(counts)]


def make_iceberg_files(counts: list[int]) -> list[IcebergDataFileInfo]:
    return [
        IcebergDataFileInfo(path=f"f{i}.parquet", record_count=c, partition={"region": "US"})
        for i, c in enumerate(counts)
    ]


def paths(shards) -> list[list[str]]:
    """Extract file paths from shards for easy comparison."""
    return [[s.file.path for s in sh.splits] for sh in shards]


def all_paths(shards) -> list[str]:
    """Flatten all paths across all shards."""
    return [s.file.path for sh in shards for s in sh.splits]


# ============================================================
# RoundRobinSplitStrategy
# ============================================================


class TestRoundRobin:
    def test_even_distribution(self):
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(make_files(8), num_workers=4)
        assert all(len(sh.splits) == 2 for sh in shards)

    def test_uneven_distribution(self):
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(make_files(9), num_workers=4)
        sizes = [len(sh.splits) for sh in shards]
        assert max(sizes) - min(sizes) <= 1

    def test_no_file_in_two_shards(self):
        files = make_files(9)
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(files, num_workers=4)
        assert sorted(all_paths(shards)) == sorted(f.path for f in files)

    def test_correct_shard_count(self):
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(make_files(9), num_workers=4)
        assert len(shards) == 4
        assert [sh.id for sh in shards] == [0, 1, 2, 3]

    def test_single_worker(self):
        files = make_files(5)
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(files, num_workers=1)
        assert len(shards) == 1
        assert len(shards[0].splits) == 5

    def test_empty_files(self):
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate([], num_workers=4)
        assert all(sh.splits == [] for sh in shards)

    def test_row_range_is_none(self):
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(make_files(4), num_workers=2)
        for sh in shards:
            for sp in sh.splits:
                assert sp.row_range is None

    def test_ignores_file_size(self):
        files = make_sized_files([1000, 1, 1, 1, 1, 1, 1, 1])
        strategy = RoundRobinSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        assert all(len(sh.splits) == 4 for sh in shards)

    def test_no_shuffle_preserves_order(self):
        files = make_files(5)
        strategy = RoundRobinSplitStrategy(shuffle=False)
        shards = strategy.generate(files, num_workers=1)
        assert [sp.file.path for sp in shards[0].splits] == [f.path for f in files]

    def test_shuffle_reproducible_same_epoch(self):
        files = make_files(9)
        strategy = RoundRobinSplitStrategy(shuffle=True, seed=42)
        s1 = strategy.generate(files, num_workers=4, epoch=0)
        s2 = strategy.generate(files, num_workers=4, epoch=0)
        assert paths(s1) == paths(s2)

    def test_shuffle_differs_across_epochs(self):
        files = make_files(9)
        strategy = RoundRobinSplitStrategy(shuffle=True, seed=42)
        s0 = strategy.generate(files, num_workers=1, epoch=0)
        s1 = strategy.generate(files, num_workers=1, epoch=1)
        assert paths(s0) != paths(s1)

    def test_shuffle_does_not_mutate_input(self):
        files = make_files(9)
        original_paths = [f.path for f in files]
        RoundRobinSplitStrategy(shuffle=True, seed=42).generate(files, num_workers=4)
        assert [f.path for f in files] == original_paths


# ============================================================
# SizeBalancedSplitStrategy
# ============================================================


class TestSizeBalanced:
    def test_balanced_by_record_count(self):
        files = make_counted_files([1000, 500, 300, 200])
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        totals = [sum(sp.file.record_count for sp in sh.splits) for sh in shards]
        assert totals[0] == totals[1]

    def test_balanced_by_file_size(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        totals = [sum(sp.file.file_size for sp in sh.splits) for sh in shards]
        assert totals[0] == totals[1]

    def test_record_count_preferred_over_file_size(self):
        files = [
            DataFileInfo(path="f0.parquet", record_count=1000, file_size=1),
            DataFileInfo(path="f1.parquet", record_count=500, file_size=999),
            DataFileInfo(path="f2.parquet", record_count=300, file_size=999),
            DataFileInfo(path="f3.parquet", record_count=200, file_size=999),
        ]
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        totals = [sum(sp.file.record_count for sp in sh.splits) for sh in shards]
        assert totals[0] == totals[1]

    def test_fallback_to_roundrobin_no_metadata(self):
        files = make_files(8)
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=4)
        assert all(len(sh.splits) == 2 for sh in shards)

    def test_no_file_in_two_shards(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        assert sorted(all_paths(shards)) == sorted(f.path for f in files)

    def test_iceberg_files_use_record_count(self):
        files = make_iceberg_files([1000, 500, 300, 200])
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        totals = [sum(sp.file.record_count for sp in sh.splits) for sh in shards]
        assert totals[0] == totals[1]

    def test_row_range_is_none(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        for sh in shards:
            for sp in sh.splits:
                assert sp.row_range is None

    def test_shuffle_reproducible(self):
        files = make_sized_files([100, 50, 30, 20, 10])
        strategy = SizeBalancedSplitStrategy(shuffle=True, seed=42)
        s1 = strategy.generate(files, num_workers=2, epoch=0)
        s2 = strategy.generate(files, num_workers=2, epoch=0)
        assert paths(s1) == paths(s2)

    def test_shuffle_does_not_mutate_input(self):
        files = make_sized_files([100, 50, 30, 20])
        original_paths = [f.path for f in files]
        SizeBalancedSplitStrategy(shuffle=True, seed=42).generate(files, num_workers=2)
        assert [f.path for f in files] == original_paths


# ============================================================
# SplitStrategy protocol
# ============================================================


def test_custom_strategy_no_inheritance():
    from torch_dataloader_utils.splits.core import Shard, Split

    class MyStrategy:
        def generate(self, files, num_workers, epoch=0) -> list[Shard]:
            return [Shard(id=i, splits=[Split(file=f) for f in files]) for i in range(num_workers)]

    shards = MyStrategy().generate(make_files(4), num_workers=2)
    assert len(shards) == 2


# ============================================================
# SizeBalancedSplitStrategy: mixed metadata
# ============================================================


class TestSizeBalancedMixedMetadata:
    def test_some_files_missing_weight_treated_as_zero(self):
        """Files with no record_count or file_size are treated as weight 0."""
        files = [
            DataFileInfo(path="f0.parquet", record_count=1000),
            DataFileInfo(path="f1.parquet", record_count=None, file_size=None),  # no weight
            DataFileInfo(path="f2.parquet", record_count=500),
        ]
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        all_file_paths = [sp.file.path for sh in shards for sp in sh.splits]
        assert sorted(all_file_paths) == ["f0.parquet", "f1.parquet", "f2.parquet"]

    def test_mixed_record_count_and_file_size_prefers_record_count(self):
        """When first file has record_count, metric is record_count for all."""
        files = [
            DataFileInfo(path="f0.parquet", record_count=1000, file_size=1),
            DataFileInfo(path="f1.parquet", record_count=1000, file_size=999),
        ]
        strategy = SizeBalancedSplitStrategy()
        shards = strategy.generate(files, num_workers=2)
        # Both files have equal record_count → one file per shard
        assert all(len(sh.splits) == 1 for sh in shards)
