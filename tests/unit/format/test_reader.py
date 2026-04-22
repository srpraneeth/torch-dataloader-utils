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


# ---------------------------------------------------------------------------
# Scenario: _parse_hive_partitions helper
# ---------------------------------------------------------------------------

def test_parse_hive_partitions_basic():
    from torch_dataloader_utils.format.reader import _parse_hive_partitions
    parts = _parse_hive_partitions("/data/region=us/year=2024/part.parquet")
    assert parts == {"region": "us", "year": "2024"}


def test_parse_hive_partitions_no_partitions():
    from torch_dataloader_utils.format.reader import _parse_hive_partitions
    parts = _parse_hive_partitions("/data/part.parquet")
    assert parts == {}


def test_parse_hive_partitions_single():
    from torch_dataloader_utils.format.reader import _parse_hive_partitions
    parts = _parse_hive_partitions("s3://bucket/dt=2024-01-01/file.parquet")
    assert parts == {"dt": "2024-01-01"}


def test_parse_hive_partitions_values_are_strings():
    from torch_dataloader_utils.format.reader import _parse_hive_partitions
    parts = _parse_hive_partitions("/data/year=2024/month=01/file.parquet")
    assert isinstance(parts["year"], str)
    assert isinstance(parts["month"], str)
    assert parts["year"] == "2024"
    assert parts["month"] == "01"


# ---------------------------------------------------------------------------
# Scenario: Hive partitioning — scanner path (pad.dataset)
# ---------------------------------------------------------------------------

def test_hive_partitioning_scanner_path(tmp_path):
    """partitioning="hive" via the scanner path attaches partition columns."""
    import pyarrow.parquet as pq

    # Build partitioned directory: data/region=us/year=2024/part.parquet
    partitioned_dir = tmp_path / "region=us" / "year=2024"
    partitioned_dir.mkdir(parents=True)
    part_file = partitioned_dir / "part.parquet"
    pq.write_table(pa.table({"value": pa.array([1, 2, 3], pa.int32())}), part_file)

    shard = _shard_from(str(part_file))
    batches = list(read_split(shard, format="parquet", partitioning="hive"))
    assert len(batches) > 0
    names = set(batches[0].schema.names)
    assert "value" in names
    assert "region" in names
    assert "year" in names
    # Partition values should be consistent
    regions = batches[0].column("region").to_pylist()
    assert all(r == "us" for r in regions)


def test_no_partitioning_does_not_inject_columns(tmp_path):
    """Default partitioning=None must not add extra columns."""
    import pyarrow.parquet as pq

    partitioned_dir = tmp_path / "region=us"
    partitioned_dir.mkdir(parents=True)
    part_file = partitioned_dir / "part.parquet"
    pq.write_table(pa.table({"value": pa.array([1, 2, 3], pa.int32())}), part_file)

    shard = _shard_from(str(part_file))
    batches = list(read_split(shard, format="parquet", partitioning=None))
    assert len(batches) > 0
    # partitioning=None: region column NOT present
    assert "region" not in batches[0].schema.names


# ---------------------------------------------------------------------------
# Scenario: Hive partitioning — row-range path
# ---------------------------------------------------------------------------

def test_hive_partitioning_row_range_path(tmp_path):
    """Row-range reads parse key=value from path and attach as constant columns."""
    partitioned_dir = tmp_path / "region=eu" / "year=2023"
    partitioned_dir.mkdir(parents=True)
    path = str(partitioned_dir / "part.parquet")
    _write_multi_rg_parquet(path, num_groups=2, rows_per_group=25)  # 50 rows

    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet", partitioning="hive"))
    assert len(batches) > 0
    names = set(batches[0].schema.names)
    assert "region" in names
    assert "year" in names
    regions = batches[0].column("region").to_pylist()
    assert all(r == "eu" for r in regions)
    years = batches[0].column("year").to_pylist()
    assert all(y == "2023" for y in years)


def test_hive_partitioning_row_range_no_partitioning(tmp_path):
    """Row-range with partitioning=None must not inject columns."""
    partitioned_dir = tmp_path / "region=eu"
    partitioned_dir.mkdir(parents=True)
    path = str(partitioned_dir / "part.parquet")
    _write_multi_rg_parquet(path, num_groups=2, rows_per_group=25)

    shard = _shard_with_row_range(path, offset=0, length=50)
    batches = list(read_split(shard, format="parquet", partitioning=None))
    assert len(batches) > 0
    assert "region" not in batches[0].schema.names


# ---------------------------------------------------------------------------
# Scenario: RowRange with no overlapping row groups → early return (line 160)
# ---------------------------------------------------------------------------

def test_row_range_beyond_file_yields_nothing(tmp_path):
    """A RowRange whose offset is past all row groups produces no batches."""
    path = str(tmp_path / "f.parquet")
    _write_multi_rg_parquet(path, num_groups=4, rows_per_group=25)  # 100 rows total
    # offset=500 is well beyond the 100 rows — no row group overlaps
    shard = _shard_with_row_range(path, offset=500, length=50)
    batches = list(read_split(shard, format="parquet"))
    assert batches == []


# ---------------------------------------------------------------------------
# Scenario: _get_arrow_filesystem — non-local path wraps fsspec (lines 206-208)
# ---------------------------------------------------------------------------

def test_get_arrow_filesystem_remote_returns_py_filesystem():
    """A non-local path (e.g. s3://) wraps the fsspec fs in pafs.PyFileSystem."""
    from unittest.mock import MagicMock, patch
    import pyarrow.fs as pafs
    from torch_dataloader_utils.format.reader import _get_arrow_filesystem

    mock_fs = MagicMock()
    mock_fs.protocol = "s3"

    mock_py_fs = MagicMock(spec=pafs.PyFileSystem)

    with patch("torch_dataloader_utils.format.reader.fsspec.url_to_fs",
               return_value=(mock_fs, "bucket/key.parquet")):
        with patch("torch_dataloader_utils.format.reader.pafs.FSSpecHandler"):
            with patch("torch_dataloader_utils.format.reader.pafs.PyFileSystem",
                       return_value=mock_py_fs):
                arrow_fs, resolved = _get_arrow_filesystem("s3://bucket/key.parquet", {})

    assert resolved == "bucket/key.parquet"
    assert arrow_fs is mock_py_fs


def test_get_arrow_filesystem_local_returns_none():
    """A local path returns (None, resolved_path) — no wrapping."""
    from torch_dataloader_utils.format.reader import _get_arrow_filesystem

    arrow_fs, resolved = _get_arrow_filesystem("/tmp/data.parquet", {})
    assert arrow_fs is None
    assert resolved == "/tmp/data.parquet"

