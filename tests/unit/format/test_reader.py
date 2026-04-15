import pytest
import pyarrow as pa
import pyarrow.compute as pc

from torch_dataloader_utils.format.reader import read_split
from torch_dataloader_utils.splits.core import DataFileInfo, FileSplit, Split


FIXTURES = __import__("pathlib").Path(__file__).parent.parent.parent / "fixtures"


def _split_from(path: str) -> Split:
    return Split(id=0, file_splits=[FileSplit(file=DataFileInfo(path=path))])


def _split_from_paths(*paths: str) -> Split:
    return Split(
        id=0,
        file_splits=[FileSplit(file=DataFileInfo(path=p)) for p in paths],
    )


# ---------------------------------------------------------------------------
# Scenario: Each format reads without error
# ---------------------------------------------------------------------------

def test_read_parquet():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_orc():
    split = _split_from(str(FIXTURES / "sample.orc"))
    batches = list(read_split(split, format="orc"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_csv():
    split = _split_from(str(FIXTURES / "sample.csv"))
    batches = list(read_split(split, format="csv"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_jsonl():
    split = _split_from(str(FIXTURES / "sample.jsonl"))
    batches = list(read_split(split, format="jsonl"))
    assert len(batches) > 0
    assert isinstance(batches[0], pa.RecordBatch)


def test_read_json_alias():
    # "json" and "jsonl" are both valid
    split = _split_from(str(FIXTURES / "sample.jsonl"))
    batches = list(read_split(split, format="json"))
    assert len(batches) > 0


# ---------------------------------------------------------------------------
# Scenario: Total row count matches fixture (5 rows)
# ---------------------------------------------------------------------------

def test_total_rows_parquet():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    total = sum(b.num_rows for b in read_split(split, format="parquet"))
    assert total == 5


def test_total_rows_csv():
    split = _split_from(str(FIXTURES / "sample.csv"))
    total = sum(b.num_rows for b in read_split(split, format="csv"))
    assert total == 5


# ---------------------------------------------------------------------------
# Scenario: Column projection
# ---------------------------------------------------------------------------

def test_column_projection():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet", columns=["feature_a", "label"]))
    assert batches[0].schema.names == ["feature_a", "label"]


def test_column_projection_single():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet", columns=["label"]))
    assert batches[0].schema.names == ["label"]


def test_no_column_projection_returns_all():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet", columns=None))
    assert set(batches[0].schema.names) == {"feature_a", "feature_b", "label"}


# ---------------------------------------------------------------------------
# Scenario: Predicate pushdown
# ---------------------------------------------------------------------------

def test_filter_pushdown():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    # feature_b values are [10, 20, 30, 40, 50] — filter keeps > 30 → [40, 50]
    batches = list(read_split(
        split, format="parquet", filters=pc.field("feature_b") > 30
    ))
    total = sum(b.num_rows for b in batches)
    assert total == 2


def test_filter_pushdown_no_matches():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(
        split, format="parquet", filters=pc.field("feature_b") > 1000
    ))
    total = sum(b.num_rows for b in batches)
    assert total == 0


# ---------------------------------------------------------------------------
# Scenario: Batch size
# ---------------------------------------------------------------------------

def test_batch_size_respected():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet", batch_size=2))
    assert all(b.num_rows <= 2 for b in batches)


def test_batch_size_larger_than_file():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    # file has 5 rows, batch_size=100 → single batch
    batches = list(read_split(split, format="parquet", batch_size=100))
    assert len(batches) == 1
    assert batches[0].num_rows == 5


# ---------------------------------------------------------------------------
# Scenario: Unsupported format
# ---------------------------------------------------------------------------

def test_unsupported_format_raises():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="avro"):
        list(read_split(split, format="avro"))


def test_unsupported_format_lists_supported():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="parquet"):
        list(read_split(split, format="avro"))


# ---------------------------------------------------------------------------
# Scenario: Files read in order
# ---------------------------------------------------------------------------

def test_files_read_in_order(tmp_path):
    import pyarrow.parquet as pq

    f1 = tmp_path / "f1.parquet"
    f2 = tmp_path / "f2.parquet"
    pq.write_table(pa.table({"val": pa.array([1, 2])}), f1)
    pq.write_table(pa.table({"val": pa.array([3, 4])}), f2)

    split = _split_from_paths(str(f1), str(f2))
    values = []
    for batch in read_split(split, format="parquet", batch_size=10):
        values.extend(batch.column("val").to_pylist())

    assert values == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Scenario: Empty split
# ---------------------------------------------------------------------------

def test_empty_split_yields_nothing():
    split = Split(id=0, file_splits=[])
    batches = list(read_split(split, format="parquet"))
    assert batches == []


# ---------------------------------------------------------------------------
# Scenario: storage_options passthrough (local — empty dict)
# ---------------------------------------------------------------------------

def test_storage_options_local():
    split = _split_from(str(FIXTURES / "sample.parquet"))
    batches = list(read_split(split, format="parquet", storage_options={}))
    assert len(batches) > 0
