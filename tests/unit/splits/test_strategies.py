from torch_dataloader_utils.splits.core import DataFileInfo, IcebergDataFileInfo
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy
from torch_dataloader_utils.splits.balanced import SizeBalancedSplitStrategy


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


def paths(splits) -> list[list[str]]:
    """Extract file paths from splits for easy comparison."""
    return [[fs.file.path for fs in s.file_splits] for s in splits]


def all_paths(splits) -> list[str]:
    """Flatten all paths across all splits."""
    return [fs.file.path for s in splits for fs in s.file_splits]


# ============================================================
# RoundRobinSplitStrategy
# ============================================================

class TestRoundRobin:

    def test_even_distribution(self):
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(make_files(8), num_workers=4)
        assert all(len(s.file_splits) == 2 for s in splits)

    def test_uneven_distribution(self):
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(make_files(9), num_workers=4)
        sizes = [len(s.file_splits) for s in splits]
        assert max(sizes) - min(sizes) <= 1

    def test_no_file_in_two_splits(self):
        files = make_files(9)
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(files, num_workers=4)
        assert sorted(all_paths(splits)) == sorted(f.path for f in files)

    def test_correct_split_count(self):
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(make_files(9), num_workers=4)
        assert len(splits) == 4
        assert [s.id for s in splits] == [0, 1, 2, 3]

    def test_single_worker(self):
        files = make_files(5)
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(files, num_workers=1)
        assert len(splits) == 1
        assert len(splits[0].file_splits) == 5

    def test_empty_files(self):
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate([], num_workers=4)
        assert all(s.file_splits == [] for s in splits)

    def test_row_range_is_none(self):
        # V1 — all FileSplits must have row_range=None
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(make_files(4), num_workers=2)
        for s in splits:
            for fs in s.file_splits:
                assert fs.row_range is None

    def test_ignores_file_size(self):
        files = make_sized_files([1000, 1, 1, 1, 1, 1, 1, 1])
        strategy = RoundRobinSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        assert all(len(s.file_splits) == 4 for s in splits)

    def test_no_shuffle_preserves_order(self):
        files = make_files(5)
        strategy = RoundRobinSplitStrategy(shuffle=False)
        splits = strategy.generate(files, num_workers=1)
        assert [fs.file.path for fs in splits[0].file_splits] == [f.path for f in files]

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
        splits = strategy.generate(files, num_workers=2)
        totals = [sum(fs.file.record_count for fs in s.file_splits) for s in splits]
        assert totals[0] == totals[1]

    def test_balanced_by_file_size(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        totals = [sum(fs.file.file_size for fs in s.file_splits) for s in splits]
        assert totals[0] == totals[1]

    def test_record_count_preferred_over_file_size(self):
        files = [
            DataFileInfo(path="f0.parquet", record_count=1000, file_size=1),
            DataFileInfo(path="f1.parquet", record_count=500,  file_size=999),
            DataFileInfo(path="f2.parquet", record_count=300,  file_size=999),
            DataFileInfo(path="f3.parquet", record_count=200,  file_size=999),
        ]
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        totals = [sum(fs.file.record_count for fs in s.file_splits) for s in splits]
        assert totals[0] == totals[1]

    def test_fallback_to_roundrobin_no_metadata(self):
        files = make_files(8)
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=4)
        assert all(len(s.file_splits) == 2 for s in splits)

    def test_no_file_in_two_splits(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        assert sorted(all_paths(splits)) == sorted(f.path for f in files)

    def test_iceberg_files_use_record_count(self):
        files = make_iceberg_files([1000, 500, 300, 200])
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        totals = [sum(fs.file.record_count for fs in s.file_splits) for s in splits]
        assert totals[0] == totals[1]

    def test_row_range_is_none(self):
        files = make_sized_files([100, 50, 30, 20])
        strategy = SizeBalancedSplitStrategy()
        splits = strategy.generate(files, num_workers=2)
        for s in splits:
            for fs in s.file_splits:
                assert fs.row_range is None

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
    from torch_dataloader_utils.splits.core import FileSplit, Split

    class MyStrategy:
        def generate(self, files, num_workers, epoch=0) -> list[Split]:
            return [Split(id=i, file_splits=[FileSplit(file=f) for f in files])
                    for i in range(num_workers)]

    splits = MyStrategy().generate(make_files(4), num_workers=2)
    assert len(splits) == 2
