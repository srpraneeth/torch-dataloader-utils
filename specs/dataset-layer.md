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

**Output formats**

| `output_format` | Numeric column type | Non-numeric column type |
|-----------------|--------------------|-----------------------|
| `"torch"` (default) | `torch.Tensor` | `list` |
| `"numpy"` | `np.ndarray` | `list` |
| `"arrow"` | `pyarrow.RecordBatch` (whole batch) | — |
| `"dict"` | `list` | `list` |

**Construction errors**

| Condition | Error |
|-----------|-------|
| `format="avro"` | `ValueError` at construction |
| `output_format="arrow"`, `collate_fn=None` | `ValueError` at construction |
| `output_format="dict"`, `collate_fn=None` | `ValueError` at construction |

**Worker assignment**

| num_workers | files | Expected |
|-------------|-------|----------|
| 0 (single process) | any | shard 0 reads all splits |
| 4 | 4 Parquet files | each worker reads exactly 1 file |
| 2 | 4 Parquet files | each worker reads 2 files, 0 rows dropped |
| 4 | 1 Parquet file (1 split) | worker 0 reads it; workers 1–3 yield nothing |

**Shuffle** — `shuffle=True, seed=42`: two loaders at same epoch produce identical row order; epoch 0 vs epoch 1 produce different order

**`set_epoch()`** — must be called in main process; splits are regenerated with `seed + epoch`; has no effect when `shuffle=False`

**`num_workers=None`** — auto-detects to `max(1, os.cpu_count() - 1)`, logged at INFO

**Column projection end-to-end** — `columns=["feature_a", "label"]` → each batch contains only those keys

**Predicate pushdown end-to-end** — `filters=pc.field("feature_b") > 30` → only matching rows returned

**Hive partitioning end-to-end** — `data/region=us/year=2024/part.parquet`, `partitioning="hive"` → batches include `region` and `year` columns

**`scan_filter` (Iceberg)** — `GreaterThan("region_id", 5)` passed to `table.scan()` → only files that could match are scanned; splits generated over pruned file set only

**`scan_filter` auto-derivation** — `filters=pc.field("row_id") >= 1875`, `scan_filter=None` → system translates to `GreaterThanOrEqual("row_id", 1875)`, logs at INFO, prunes files

**Untranslatable filter fallback** — `filters=pc.field("a") >= pc.field("b")` (field vs field) → translation returns None, full file scan, filter applied row-level, DEBUG log emitted

**`split_bytes` sub-file splitting** — large Parquet file with 10 row groups, `split_bytes="10KiB"` → multiple `FileSplit`s with disjoint `RowRange`s produced

**`split_rows`** — multiple Parquet files, `split_rows=5000` → each chunk covers ≤ 5000 rows aligned to row group boundaries
