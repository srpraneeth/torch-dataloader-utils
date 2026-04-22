# Spec: Dataset Layer

## Core Principle

The dataset layer has one responsibility — connect the filesystem, format, and split layers
into a `torch.utils.data.IterableDataset` and expose a `create_dataloader()` entry point.

```
create_dataloader()
    → discover_files()          ← filesystem layer
    → auto-select strategy      ← split layer
    → StructuredDataset(files, splits_config, read_config)
    → DataLoader(dataset, batch_size=None, num_workers=N)
```

---

## Requirements

### Construction `[v1]`
The system SHALL accept the following parameters at construction:

**StructuredDataset:**
- `path: str` — passed to `discover_files()`
- `format: str` — parquet, orc, csv, json, jsonl
- `batch_size: int` — default 1024
- `columns: list[str] | None` — column projection
- `filters: pc.Expression | None` — row-level predicate pushdown via pyarrow
- `shuffle: bool` — default False
- `shuffle_seed: int` — default 42
- `split_bytes: int | str | None` — target chunk size (e.g. `"128MiB"`, `10_485_760`). None = 128 MiB default
- `split_rows: int | None` — target rows per chunk (overrides split_bytes when set)
- `split_strategy: SplitStrategy | None` — None = auto-select
- `num_workers: int | None` — None = auto-detect via `max(1, os.cpu_count() - 1)`
- `output_format: str` — torch | numpy | arrow | dict, default "torch"
- `storage_options: dict | None` — forwarded to fsspec and read_split
- `collate_fn: Callable | None` — forwarded to DataLoader
- `partitioning: str | None` — None = no partitioning decoding; `"hive"` = decode Hive-style directory partitions

**IcebergDataset** (all StructuredDataset params except `path`/`format`, plus):
- `table: str` — fully qualified table identifier (e.g. `"my_db.my_table"` or `"ns.db.table"`)
- `catalog_config: dict` — forwarded directly to `pyiceberg.catalog.load_catalog()`
- `snapshot_id: int | None` — None = current snapshot; provide for time travel
- `scan_filter` — pyiceberg `BooleanExpression` pushed into `table.scan(row_filter=...)` at plan_files() time. Prunes entire partitions and files before splits are generated. Use alongside `filters` for belt-and-suspenders filtering.

The system SHALL raise `ValueError` for unsupported `format` values at construction time.
The system SHALL raise `ValueError` for unsupported `output_format` values at construction time.
The system SHALL raise `ValueError` when `output_format` is `"arrow"` or `"dict"` and `collate_fn` is None.
`IcebergDataset` SHALL raise `FileNotFoundError` when no data files are found after scan.

### File Discovery `[v1]`
File discovery SHALL happen once at `create_dataloader()` time via `discover_files()`.
The discovered `list[DataFileInfo]` SHALL be stored on the dataset instance.

### Iceberg Scan-Level Predicate Pushdown `[v1]`
When `scan_filter` is provided, it SHALL be passed to `table.scan(row_filter=scan_filter)`.
This causes `plan_files()` to exclude partitions and files that cannot satisfy the predicate.
The `scan_filter` SHALL be applied before split generation — splits are computed only over surviving files.
`scan_filter` accepts native pyiceberg expressions (`GreaterThan`, `LessThan`, etc.) — NOT pyarrow expressions.
`filters` (pyarrow) and `scan_filter` (pyiceberg) MAY be used together for belt-and-suspenders filtering.

### Scan-Filter Auto-Derivation `[v1]`
When `scan_filter=None` and `filters` is provided, the system SHALL attempt to translate the `pc.Expression` into a pyiceberg `BooleanExpression` and use it as `scan_filter`.

Supported translations:
- Comparisons with scalar literals: `>=` → `GreaterThanOrEqual`, `>` → `GreaterThan`, `<=` → `LessThanOrEqual`, `<` → `LessThan`, `==` → `EqualTo`, `!=` → `NotEqualTo`
- Literal types: integers, floats, strings
- Compound: `&` → `And`, `|` → `Or` (arbitrarily nested)

When translation succeeds, the system SHALL log at `INFO`: `"Auto-derived scan_filter: pc.Expression <X> → pyiceberg <Y>"`.
When translation fails (unsupported expression or pyiceberg not installed), the system SHALL log at `DEBUG` and fall back to row-level filtering only — no error is raised.
When `scan_filter` is explicitly provided alongside `filters`, the explicit value SHALL be used without attempting translation.

### Split Generation `[v1]`
Splits SHALL be generated once at construction time and cached on `_splits`.
When `shuffle=True`, splits SHALL be regenerated each time `set_epoch(n)` is called using `shuffle_seed + epoch`.
`set_epoch()` SHALL be called in the main process before each epoch.

### Strategy Auto-Selection `[v1]`
When `split_strategy=None` the system SHALL auto-select:
- Non-empty file list → `TargetSizeSplitStrategy`
- Empty file list → `RoundRobinSplitStrategy`

`TargetSizeSplitStrategy` behaviour:
- Parquet files: read row group metadata from file footer (no data scan), pack consecutive row groups into chunks targeting `split_bytes`. Each chunk becomes a `FileSplit` with a `RowRange`.
- Non-Parquet files: treated as a single unsplittable chunk (no footer metadata).
- `split_rows` overrides `split_bytes` when both are set.
- Chunks are distributed across workers using a greedy min-heap (LPT scheduling) — minimises maximum worker load for unequal chunk sizes.
- Default `target_bytes` = 128 MiB.

The system SHALL log the selected strategy at `INFO` level.

### Worker Assignment `[v1]`
The system SHALL use `torch.utils.data.get_worker_info()` to determine the current worker index.
When `num_workers=0` (single process) the system SHALL use shard index 0.
Workers with `worker_id >= len(shards)` SHALL yield nothing (more workers than shards).

### Iteration `[v1]`
Each worker SHALL call `read_split()` with its assigned `Shard`.
Each worker SHALL yield one item per `RecordBatch` after output format conversion.
The system SHALL NOT buffer or accumulate batches — yield immediately.

When `has_deletes=True` (Iceberg tables with position delete files), each worker SHALL reconnect to the catalog and use `pyiceberg.io.pyarrow.ArrowScan` to apply position deletes. Sub-file row-range splitting is disabled in this path — file-granularity splits only.

### Output Format Conversion `[v1]`
The system SHALL convert each `pyarrow.RecordBatch` to the requested `output_format`:

| output_format | Type | When to use |
|---------------|------|-------------|
| `"torch"`  | `dict[str, torch.Tensor \| list]` | PyTorch training (default) |
| `"numpy"`  | `dict[str, np.ndarray \| list]` | sklearn, lighter weight |
| `"arrow"`  | `pyarrow.RecordBatch` | zero conversion, custom collate |
| `"dict"`   | `dict[str, list]` | debugging, string columns |

For `"torch"`, the system SHALL use best-effort conversion:
- Numeric columns (int, float, bool) → `torch.from_numpy(col.to_numpy())` (zero-copy numpy→tensor)
- Non-numeric columns (string, binary, etc.) → passed through as Python `list`

For `"numpy"`, the system SHALL use best-effort conversion:
- Numeric columns → `col.to_numpy()`
- Non-numeric columns → passed through as Python `list`

For `"arrow"`, the `RecordBatch` is yielded as-is — no conversion.
For `"dict"`, each column is converted via `col.to_pylist()`.

### `num_workers` Auto-Detection `[v1]`
When `num_workers=None` the system SHALL use `max(1, os.cpu_count() - 1)`.
The system SHALL log the resolved value at `INFO` level.
`num_workers=0` SHALL mean single-process mode (PyTorch convention).

### `create_dataloader()` `[v1]`
The system SHALL provide a `create_dataloader()` classmethod as the primary entry point.
It SHALL perform file discovery, strategy selection, and return a `(DataLoader, dataset)` tuple.
It SHALL construct `DataLoader(dataset, batch_size=None, num_workers=N, collate_fn=collate_fn)`.
`batch_size=None` is always passed — Arrow owns batching, not DataLoader.

### Logging `[v1]`
The system SHALL log at `INFO`: path/table, format, num_workers, split strategy selected, file count, per-file breakdown, inferred schema.
The system SHALL log at `INFO` per worker: shard id, number of splits, per-split file name + row range + row count.
The system SHALL log at `DEBUG`: worker index, file paths in split.

---

## Interface

```python
class StructuredDataset(IterableDataset):
    def __init__(
        self,
        files: list[DataFileInfo],
        format: str,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
        partitioning: str | None = None,
    ) -> None: ...

    def set_epoch(self, epoch: int) -> None: ...
    def __iter__(self) -> Iterator: ...

    @classmethod
    def create_dataloader(
        cls,
        path: str,
        format: str,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
        partitioning: str | None = None,
    ) -> tuple[DataLoader, "StructuredDataset"]: ...


class IcebergDataset(IterableDataset):
    def __init__(
        self,
        table: str,
        catalog_config: dict,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        scan_filter: Any | None = None,
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        collate_fn: Callable | None = None,
    ) -> None: ...

    def set_epoch(self, epoch: int) -> None: ...
    def __iter__(self) -> Iterator: ...

    @classmethod
    def create_dataloader(
        cls,
        table: str,
        catalog_config: dict,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        scan_filter: Any | None = None,
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        collate_fn: Callable | None = None,
    ) -> tuple[DataLoader, "IcebergDataset"]: ...
```

---

## Scenarios

#### Scenario: Single-process iteration
- GIVEN a directory of Parquet files and `num_workers=0`
- WHEN the DataLoader iterates
- THEN all rows are yielded as `dict[str, torch.Tensor]`

#### Scenario: Multi-worker iteration
- GIVEN 4 Parquet files and `num_workers=4`
- WHEN the DataLoader iterates
- THEN each worker reads exactly 1 file, all rows are returned across workers

#### Scenario: Output format torch
- GIVEN `output_format="torch"` and a file with numeric and string columns
- WHEN a batch is yielded
- THEN numeric columns are `torch.Tensor` and string columns are `list`

#### Scenario: Output format numpy
- GIVEN `output_format="numpy"` and a file with numeric and string columns
- WHEN a batch is yielded
- THEN numeric columns are `np.ndarray` and string columns are `list`

#### Scenario: Output format arrow
- GIVEN `output_format="arrow"`
- WHEN a batch is yielded
- THEN it is a `pyarrow.RecordBatch`

#### Scenario: Output format dict
- GIVEN `output_format="dict"`
- WHEN a batch is yielded
- THEN it is a `dict[str, list]`

#### Scenario: arrow output without collate_fn raises at construction
- GIVEN `output_format="arrow"` and `collate_fn=None`
- WHEN `StructuredDataset` is constructed
- THEN a `ValueError` is raised immediately

#### Scenario: Shuffle reproducibility
- GIVEN `shuffle=True` and `shuffle_seed=42`
- WHEN two DataLoaders iterate with the same epoch
- THEN the file order is identical

#### Scenario: Epoch reshuffling
- GIVEN `shuffle=True`
- WHEN `set_epoch(0)` and `set_epoch(1)` are called
- THEN the file order differs between epochs

#### Scenario: Column projection end-to-end
- GIVEN `columns=["feature_a", "label"]`
- WHEN the DataLoader iterates
- THEN each batch contains only `feature_a` and `label` keys

#### Scenario: Predicate pushdown end-to-end
- GIVEN `filters=pc.field("feature_b") > 30`
- WHEN the DataLoader iterates
- THEN only rows matching the filter are returned

#### Scenario: unsupported format raises at construction
- GIVEN `format="avro"`
- WHEN `create_dataloader()` is called
- THEN a `ValueError` is raised

#### Scenario: num_workers=None auto-detects
- GIVEN `num_workers=None`
- WHEN `create_dataloader()` is called
- THEN num_workers is set to `max(1, os.cpu_count() - 1)` and logged

#### Scenario: Hive partitioning end-to-end
- GIVEN a partitioned directory `data/region=us/year=2024/part.parquet`
- WHEN `create_dataloader(path, format="parquet", partitioning="hive")` is called
- THEN each batch includes `region="us"` and `year="2024"` columns alongside data columns

#### Scenario: scan_filter prunes files at plan_files() level
- GIVEN an IcebergDataset with `scan_filter=GreaterThan("region_id", 5)`
- WHEN `_resolve_files` is called
- THEN `table.scan(row_filter=GreaterThan("region_id", 5))` is called
- AND only files that could contain matching rows are returned by `plan_files()`
- AND splits are generated over the pruned file set only

#### Scenario: scan_filter=None uses full scan
- GIVEN an IcebergDataset with `scan_filter=None`
- WHEN `_resolve_files` is called
- THEN `table.scan(snapshot_id=...)` is called without `row_filter`

#### Scenario: split_bytes triggers sub-file splitting
- GIVEN a large Parquet file with 10 row groups and `split_bytes="10KiB"`
- WHEN splits are generated
- THEN multiple FileSplits with RowRange are produced for the single file
- AND each RowRange covers a disjoint subset of row groups

#### Scenario: split_rows produces row-count-balanced splits
- GIVEN multiple Parquet files and `split_rows=5000`
- WHEN splits are generated
- THEN each chunk covers at most 5000 rows (aligned to row group boundaries)

#### Scenario: scan_filter auto-derived from filters
- GIVEN `filters=pc.field("row_id") >= 1875` and `scan_filter=None`
- WHEN `IcebergDataset` is constructed
- THEN the system translates `filters` to `GreaterThanOrEqual("row_id", 1875)`
- AND `_resolve_files` is called with that derived iceberg expression as `scan_filter`
- AND an INFO log is emitted showing both the pc.Expression and the derived pyiceberg expression
- AND splits are generated only over the pruned file set

#### Scenario: untranslatable filters falls back to row-level only
- GIVEN `filters=pc.field("a") >= pc.field("b")` (field vs field — no scalar literal) and `scan_filter=None`
- WHEN `IcebergDataset` is constructed
- THEN translation returns None, `_resolve_files` is called without `scan_filter`
- AND a DEBUG log is emitted explaining the fallback
- AND all files are scanned; the filter is applied row-level inside workers
