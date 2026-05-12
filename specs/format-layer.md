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
The system SHALL read each file fully when `FileSplit.row_range` is `None`.
The system SHALL use `pyarrow.dataset.dataset()` as the reading backend for whole-file reads.
The system SHALL pass `filesystem` to `pyarrow.dataset.dataset()` to support fsspec backends.
The system SHALL reconstruct the fsspec filesystem from the file path and `storage_options` at read time — not accept a filesystem object as input.

### Parquet Row-Range Reading `[v2]`
When `FileSplit.row_range` is set and the format is `parquet`, the system SHALL use `pq.ParquetFile.read_row_groups()` to read only the assigned row groups (true random access — no full scan).
The system SHALL walk cumulative row group row counts to resolve which row groups are covered by the `RowRange`.
Partition columns (Hive path segments) SHALL be parsed from the file path and attached as constant columns on each yielded batch, matching the whole-file scanner path behaviour.

### ORC Row-Range Reading `[v2]`
When `FileSplit.row_range` is set and the format is `orc`, the system SHALL use `ORCFile.read_stripe()` to read only the assigned stripes (true random access — no full scan).
The system SHALL derive which stripe indices are assigned from the `RowRange` (offset and length in approximate rows, aligned to stripe boundaries at generation time).

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

**Format reading**

| Format | Expected |
|--------|----------|
| `parquet` | All rows from all files in split yielded as `RecordBatch` |
| `orc` | Rows yielded as `RecordBatch` |
| `csv` | Rows yielded as `RecordBatch` |
| `json` / `jsonl` | Rows yielded as `RecordBatch` |
| `avro` (unsupported) | `ValueError` naming the format and listing supported ones |

**Column projection** — `columns=["a", "b"]` on file with `[a, b, c]` → only `a`, `b` present in each batch

**Predicate pushdown** — `filters=pc.field("value") > 50` on values 1–100 → only rows where value > 50 returned

**Batch size** — `batch_size=10` on 100-row file → each `RecordBatch` has ≤ 10 rows; `batch_size=100` on 5-row file → single batch of 5

**File order** — split with `[f1, f2]` → all rows from `f1` yielded before any rows from `f2`

**Empty split** — split with no `FileSplit`s → no batches yielded, no error raised

**Hive partitioning — scanner path** — file at `data/region=us/year=2024/part.parquet`, `partitioning="hive"` → batches include `region="us"` and `year="2024"` as extra columns

**Hive partitioning — row-range path** — large Parquet file at `data/region=eu/part.parquet` read via `RowRange` → batches include `region="eu"` constant column

**ORC row-range reading** — ORC file with 4 stripes, `RowRange(offset=stripe2_start, length=2_stripes_rows)` → only stripes 2 and 3 read; no rows from stripes 0 or 1 returned

**No partitioning** — `partitioning=None` (default) → partition-encoded directory values NOT injected as columns

**Storage options local** — `storage_options={}` on local file → reads successfully, no error
