import pytest
import pyarrow as pa
import pyarrow.compute as pc

from torch_dataloader_utils.format.reader import read_split
from torch_dataloader_utils.splits.core import DataFileInfo, Shard, Split


FIXTURES = __import__("pathlib").Path(__file__).parent.parent.parent / "fixtures"


def _shard_from(path: str) -> Shard:
    return Shard(id=0, splits=[Split(file=DataFileInfo(path=path))])


def _shard_from_paths(*paths: str) -> Shard:
    return Shard(
        id=0,
        splits=[Split(file=DataFileInfo(path=p)) for p in paths],
    )


# ---------------------------------------------------------------------------
# Scenario: Each format reads without error
# ---------------------------------------------------------------------------

def test_read_parquet():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_orc():
    shard = _shard_from(str(FIXTURES / "sample.orc"))
    batches = list(read_split(shard, format="orc"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_csv():
    shard = _shard_from(str(FIXTURES / "sample.csv"))
    batches = list(read_split(shard, format="csv"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_jsonl():
    shard = _shard_from(str(FIXTURES / "sample.jsonl"))
    batches = list(read_split(shard, format="jsonl"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_json_alias():
    # "json" and "jsonl" are both valid
    shard = _shard_from(str(FIXTURES / "sample.jsonl"))
    batches = list(read_split(shard, format="json"))
    assert len(batches) > 0


# ---------------------------------------------------------------------------
# Scenario: Total row count matches fixture (5 rows)
# ---------------------------------------------------------------------------

def test_total_rows_parquet():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    total = sum(b.num_rows for b in read_split(shard, format="parquet"))
    assert total == 5


def test_total_rows_csv():
    shard = _shard_from(str(FIXTURES / "sample.csv"))
    total = sum(b.num_rows for b in read_split(shard, format="csv"))
    assert total == 5


# ---------------------------------------------------------------------------
# Scenario: Column projection
# ---------------------------------------------------------------------------

def test_column_projection():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet", columns=["feature_a", "label"]))
    assert batches[0].schema.names == ["feature_a", "label"]


def test_column_projection_single():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet", columns=["label"]))
    assert batches[0].schema.names == ["label"]


def test_no_column_projection_returns_all():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet", columns=None))
    assert set(batches[0].schema.names) == {"feature_a", "feature_b", "label"}


# ---------------------------------------------------------------------------
# Scenario: Predicate pushdown
# ---------------------------------------------------------------------------

def test_filter_pushdown():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    # feature_b values are [10, 20, 30, 40, 50] — filter keeps > 30 → [40, 50]
    batches = list(read_split(
        shard, format="parquet", filters=pc.field("feature_b") > 30
    ))
    total = sum(b.num_rows for b in batches)
    assert total == 2


def test_filter_pushdown_no_matches():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(
        shard, format="parquet", filters=pc.field("feature_b") > 1000
    ))
    total = sum(b.num_rows for b in batches)
    assert total == 0


# ---------------------------------------------------------------------------
# Scenario: Batch size
# ---------------------------------------------------------------------------

def test_batch_size_respected():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet", batch_size=2))
    assert all(b.num_rows <= 2 for b in batches)


def test_batch_size_larger_than_file():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    # file has 5 rows, batch_size=100 → single batch
    batches = list(read_split(shard, format="parquet", batch_size=100))
    assert len(batches) == 1
    assert batches[0].num_rows == 5


# ---------------------------------------------------------------------------
# Scenario: Unsupported format
# ---------------------------------------------------------------------------

def test_unsupported_format_raises():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="avro"):
        list(read_split(shard, format="avro"))


def test_unsupported_format_lists_supported():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="parquet"):
        list(read_split(shard, format="avro"))


# ---------------------------------------------------------------------------
# Scenario: Files read in order
# ---------------------------------------------------------------------------

def test_files_read_in_order(tmp_path):
    import pyarrow.parquet as pq

    f1 = tmp_path / "f1.parquet"
    f2 = tmp_path / "f2.parquet"
    pq.write_table(pa.table({"val": pa.array([1, 2])}), f1)
    pq.write_table(pa.table({"val": pa.array([3, 4])}), f2)

    shard = _shard_from_paths(str(f1), str(f2))
    values = []
    for batch in read_split(shard, format="parquet", batch_size=10):
        values.extend(batch.column("val").to_pylist())

    assert values == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Scenario: Empty shard
# ---------------------------------------------------------------------------

def test_empty_shard_yields_nothing():
    shard = Shard(id=0, splits=[])
    batches = list(read_split(shard, format="parquet"))
    assert batches == []


# ---------------------------------------------------------------------------
# Scenario: storage_options passthrough (local — empty dict)
# ---------------------------------------------------------------------------

def test_storage_options_local():
    shard = _shard_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(shard, format="parquet", storage_options={}))
    assert len(batches) > 0


# ---------------------------------------------------------------------------
# Scenario: _read_parquet_row_range branches — columns, filters, both, neither
# ---------------------------------------------------------------------------

def _write_multi_rg_parquet(path, num_groups: int = 4, rows_per_group: int = 25):
    """Write a Parquet file with multiple row groups for row-range testing."""
    import pyarrow.parquet as pq
    tables = [
        pa.table({
            "a": pa.array(list(range(i * rows_per_group, (i + 1) * rows_per_group)), pa.int32()),
            "b": pa.array([float(x % 10) for x in range(i * rows_per_group, (i + 1) * rows_per_group)], pa.float32()),
            "c": pa.array(["x"] * rows_per_group),
        })
        for i in range(num_groups)
    ]
    writer = pq.ParquetWriter(path, tables[0].schema)
    for t in tables:
        writer.write_table(t)
    writer.close()


def _shard_with_row_range(path: str, offset: int, length: int) -> "Shard":
    from torch_dataloader_utils.splits.core import RowRange
    return Shard(id=0, splits=[Split(
        file=DataFileInfo(path=path),
        row_range=RowRange(offset=offset, length=length),
    )])


def test_row_range_no_filter_no_columns(tmp_path):
    """Neither filter nor columns — all columns returned, all rows in range."""
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)  # 100 rows total
    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet"))
    total = sum(b.num_rows for b in batches)
    assert total == 50
    assert set(batches[0].schema.names) == {"a", "b", "c"}


def test_row_range_columns_only(tmp_path):
    """Column projection with no filter — only requested columns returned."""
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)
    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet", columns=["a"]))
    assert batches[0].schema.names == ["a"]
    assert sum(b.num_rows for b in batches) == 50


def test_row_range_filter_only(tmp_path):
    """Filter with no column projection — all columns present, rows filtered."""
    import pyarrow.compute as pc
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)
    # rows 0–49; column a has values 0–49; keep a >= 25 → 25 rows
    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet", filters=pc.field("a") >= 25))
    total = sum(b.num_rows for b in batches)
    assert total == 25
    assert "b" in batches[0].schema.names  # all columns present


def test_row_range_filter_and_columns(tmp_path):
    """Both filter and columns — filter applied first, then projected down."""
    import pyarrow.compute as pc
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)
    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(
        shard, format="parquet",
        columns=["a"], filters=pc.field("a") >= 25,
    ))
    total = sum(b.num_rows for b in batches)
    assert total == 25
    assert batches[0].schema.names == ["a"]
    assert "b" not in batches[0].schema.names


def test_row_range_filter_eliminates_all_rows(tmp_path):
    """Filter that matches zero rows in the range yields no batches."""
    import pyarrow.compute as pc
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)
    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet", filters=pc.field("a") > 9999))
    assert sum(b.num_rows for b in batches) == 0


def test_row_range_non_zero_offset(tmp_path):
    """RowRange with non-zero offset reads only the correct slice."""
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)  # rows 0–99
    # Read rows 50–99 (second half)
    shard = _shard_with_row_range(path, offset=50, length=50)
    batches = list(read_split(shard, format="parquet", columns=["a"]))
    values = []
    for b in batches:
        values.extend(b.column("a").to_pylist())
    assert len(values) == 50
    assert min(values) == 50
    assert max(values) == 99
