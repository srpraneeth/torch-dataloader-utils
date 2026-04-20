from torch_dataloader_utils.splits.core import (
    DataFileInfo,
    IcebergDataFileInfo,
    RowRange,
    Shard,
    Split,
)


def test_shard_defaults():
    s = Shard(id=0)
    assert s.splits == []
    assert s.row_count is None
    assert s.size_bytes is None


def test_shard_with_splits():
    sp = [Split(file=DataFileInfo(path="a.parquet"))]
    s = Shard(id=1, splits=sp)
    assert s.id == 1
    assert len(s.splits) == 1


def test_datafileinfo_defaults():
    f = DataFileInfo(path="s3://bucket/f1.parquet")
    assert f.path == "s3://bucket/f1.parquet"
    assert f.file_size is None
    assert f.record_count is None


def test_datafileinfo_with_metadata():
    f = DataFileInfo(path="f1.parquet", file_size=1024, record_count=500)
    assert f.file_size == 1024
    assert f.record_count == 500


def test_split_defaults():
    f = DataFileInfo(path="f1.parquet")
    sp = Split(file=f)
    assert sp.file is f
    assert sp.row_range is None   # full file read


def test_split_with_row_range():
    f = DataFileInfo(path="f1.parquet", record_count=1_000_000)
    sp = Split(file=f, row_range=RowRange(offset=0, length=250_000))
    assert sp.row_range.offset == 0
    assert sp.row_range.length == 250_000


def test_row_range():
    rr = RowRange(offset=250_000, length=250_000)
    assert rr.offset == 250_000
    assert rr.length == 250_000


def test_iceberg_datafileinfo_extends_datafileinfo():
    f = IcebergDataFileInfo(
        path="s3://bucket/f1.parquet",
        file_size=1024,
        record_count=500,
        partition={"region": "US", "date": "2024-01-01"},
        snapshot_id=12345,
    )
    assert f.path == "s3://bucket/f1.parquet"
    assert f.file_size == 1024
    assert f.record_count == 500
    assert f.partition == {"region": "US", "date": "2024-01-01"}
    assert f.snapshot_id == 12345


def test_iceberg_datafileinfo_is_datafileinfo():
    f = IcebergDataFileInfo(path="f1.parquet")
    assert isinstance(f, DataFileInfo)
