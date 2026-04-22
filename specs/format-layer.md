# Spec: Format Layer

## Core Principle

The format layer has one responsibility — accept a `Split` and yield `pyarrow.RecordBatch` objects.
It knows nothing about workers, splits assignment, or output conversion.

```
Split (list[FileSplit])  +  read options
            ↓
   pyarrow.dataset
            ↓
Iterator[pyarrow.RecordBatch]
```

---

## Requirements

### Supported Formats `[v1]`
The system SHALL support the following formats via `pyarrow.dataset`:
- `parquet`
- `orc`
- `csv`
- `json` / `jsonl`

The system SHALL raise a clear `ValueError` when an unsupported format is passed.

### Reading `[v1]`
The system SHALL accept a `Split` and yield `pyarrow.RecordBatch` objects.
The system SHALL read files in the order they appear in `split.file_splits`.
The system SHALL read each file fully when `FileSplit.row_range` is `None` (V1).
The system SHALL use `pyarrow.dataset.dataset()` as the reading backend.
The system SHALL pass `filesystem` to `pyarrow.dataset.dataset()` to support fsspec backends.
The system SHALL reconstruct the fsspec filesystem from the file path and `storage_options` at read time — not accept a filesystem object as input.

### Column Projection `[v1]`
The system SHALL accept an optional `columns` parameter (`list[str] | None`).
When `columns` is provided the system SHALL pass it to `scanner.to_batches()`.
When `columns` is `None` all columns SHALL be returned.

### Predicate Pushdown `[v1]`
The system SHALL accept an optional `filters` parameter (`pyarrow.compute.Expression | None`).
When `filters` is provided the system SHALL pass it to `pyarrow.dataset.dataset().scanner()`.
When `filters` is `None` no filtering SHALL be applied.
Predicate pushdown SHALL be delegated entirely to `pyarrow` — no filter interpretation in this layer.

### Batch Size `[v1]`
The system SHALL accept a `batch_size` parameter (`int`).
The system SHALL pass `batch_size` to `scanner.to_batches()` as `batch_size`.
The system SHALL default `batch_size` to `1024` when not specified.

### Storage Options `[v1]`
The system SHALL accept `storage_options: dict | None`.
The system SHALL reconstruct the fsspec filesystem via `fsspec.url_to_fs()` at read time.
The system SHALL pass the fsspec filesystem to `pyarrow.dataset.dataset()`.

### Hive Partitioning `[v1]`
The system SHALL accept an optional `partitioning: str | None` parameter.
When `partitioning="hive"` the system SHALL pass `partitioning="hive"` to `pyarrow.dataset.dataset()` so PyArrow decodes directory-encoded partition columns (e.g. `year=2024/month=01/`) and adds them as columns in each returned batch.
When `partitioning` is `None` (default) no partitioning decoding is applied.

For the row-range read path (`_read_parquet_row_range`), PyArrow's `pq.ParquetFile` does not decode partitioning automatically. The system SHALL parse `key=value` segments from the file path and attach them as constant columns to each yielded batch so the caller gets consistent columns on both read paths.

Partition column values SHALL be returned as strings — the caller is responsible for casting to the desired type.

### Logging `[v1]`
The system SHALL log at `DEBUG` level: file path being read and format.
The system SHALL log at `DEBUG` level: number of batches yielded per file.

---

## Interface

```python
def read_split(
    split: Split,
    format: str,
    batch_size: int = 1024,
    columns: list[str] | None = None,
    filters: pc.Expression | None = None,
    storage_options: dict | None = None,
    partitioning: str | None = None,
) -> Iterator[pa.RecordBatch]:
    ...
```

---

## Scenarios

#### Scenario: Read Parquet files
- GIVEN a Split with 2 Parquet FileSplits
- WHEN `read_split(split, format="parquet")` is called
- THEN all rows from both files are yielded as RecordBatches

#### Scenario: Read ORC files
- GIVEN a Split with 1 ORC FileSplit
- WHEN `read_split(split, format="orc")` is called
- THEN rows are yielded as RecordBatches

#### Scenario: Read CSV files
- GIVEN a Split with 1 CSV FileSplit
- WHEN `read_split(split, format="csv")` is called
- THEN rows are yielded as RecordBatches

#### Scenario: Read JSON/JSONL files
- GIVEN a Split with 1 JSONL FileSplit
- WHEN `read_split(split, format="json")` is called
- THEN rows are yielded as RecordBatches

#### Scenario: Column projection
- GIVEN a Parquet file with columns [a, b, c] and `columns=["a", "b"]`
- WHEN `read_split(split, format="parquet", columns=["a", "b"])` is called
- THEN only columns a and b are present in each RecordBatch

#### Scenario: Predicate pushdown
- GIVEN a Parquet file with rows where value in [1..100]
- WHEN `read_split(split, format="parquet", filters=pc.field("value") > 50)` is called
- THEN only rows with value > 50 are returned

#### Scenario: Batch size respected
- GIVEN a file with 100 rows and `batch_size=10`
- WHEN `read_split(split, format="parquet", batch_size=10)` is called
- THEN each RecordBatch has at most 10 rows

#### Scenario: Unsupported format
- GIVEN `format="avro"`
- WHEN `read_split(split, format="avro")` is called
- THEN a `ValueError` is raised naming the unsupported format and listing supported ones

#### Scenario: Files read in order
- GIVEN a Split with files [f1.parquet, f2.parquet]
- WHEN `read_split` is called
- THEN all rows from f1 are yielded before any rows from f2

#### Scenario: Empty split
- GIVEN a Split with no FileSplits
- WHEN `read_split` is called
- THEN no RecordBatches are yielded — no error raised

#### Scenario: Hive partitioning — scanner path
- GIVEN a directory with structure `data/region=us/year=2024/part.parquet`
- WHEN `read_split(split, format="parquet", partitioning="hive")` is called
- THEN each batch includes `region` and `year` columns with values `"us"` and `"2024"`

#### Scenario: Hive partitioning — row-range path
- GIVEN a large Parquet file at `data/region=eu/part.parquet` read via row-range sub-split
- WHEN `read_split(split, format="parquet", partitioning="hive")` is called
- THEN each batch includes a `region` column with value `"eu"`

#### Scenario: No partitioning (default)
- GIVEN `partitioning=None`
- WHEN `read_split` is called
- THEN partition columns are NOT injected into the output batches
