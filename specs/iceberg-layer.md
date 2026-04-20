# Spec: Iceberg Layer

## Core Principle

`IcebergDataset` resolves an Iceberg table to a list of data file URIs using
`pyiceberg`, then delegates all reading to `StructuredDataset`. Iceberg is a
table format on top of Parquet/ORC — after resolution the read path is identical.

```
IcebergDataset.create_dataloader(table=..., catalog_config=...)
        ↓
pyiceberg: connect to catalog → load table → scan snapshot → data file list
        ↓
list[IcebergDataFileInfo]  (path, file_size, record_count, partition, snapshot_id)
        ↓
StructuredDataset(files=..., format="parquet", ...)
        ↓
DataLoader  (same output as file-based path)
```

---

## Requirements

### Catalog Connection `[v1]`
`IcebergDataset` SHALL connect to a catalog via `pyiceberg` using `catalog_config`.
`catalog_config` SHALL be passed directly to `pyiceberg.catalog.load_catalog()` — no interpretation.
Supported catalog types (delegated to `pyiceberg`): `rest`, `glue`, `hive`, `jdbc`.
A missing `pyiceberg` dependency SHALL raise `ImportError` with the install command.

### Table Resolution `[v1]`
`IcebergDataset` SHALL resolve a dot-separated table identifier (`"db.table"` or `"catalog.db.table"`).
`IcebergDataset` SHALL support `snapshot_id` for time travel — when provided, scan that snapshot.
When `snapshot_id=None` the current snapshot is used.

### File List Construction `[v1]`
`IcebergDataset` SHALL build `list[IcebergDataFileInfo]` from the Iceberg scan result.
Each entry SHALL populate: `path`, `file_size`, `record_count`, `partition`, `snapshot_id`.
`record_count` from Iceberg column statistics SHALL be used for split balancing.

### Delete File Handling `[v1]`

#### Problem
Iceberg supports row-level deletes via **position delete files** and **equality delete files**. Reading Parquet data files directly (as `StructuredDataset` does) bypasses these delete files entirely — deleted rows would be returned to the caller. `scan.plan_files()` returns `FileScanTask` objects that include `delete_files` alongside the data file, but nothing enforces their application.

#### Solution — Per-task ArrowScan
`IcebergDataset` inspects every `FileScanTask` at construction time.

- If **no tasks have delete files** (`has_deletes=False`): use the direct pyarrow reader fast path — same as `StructuredDataset`. Sub-file row-range splitting via `TargetSizeSplitStrategy` is active.
- If **any task has delete files** (`has_deletes=True`): switch to `pyiceberg.io.pyarrow.ArrowScan.to_record_batches(tasks=[task])` per file. ArrowScan reads the position delete files and removes the deleted rows before yielding batches. Multi-worker file-level parallelism is preserved — each worker owns a split of tasks and reads them sequentially through ArrowScan.

#### Limitations
- **Equality deletes** are not supported by pyiceberg (`NotImplementedError`). Compact the table with your query engine to convert equality deletes to position deletes before training.
- **Sub-file row-range splitting is disabled** when delete files are present. Position delete file offsets reference absolute row positions in the original data file; applying them to a sub-range slice would produce incorrect results.
- **Partition pruning** via `pc.Expression` is not applied at the Iceberg scan level — pyiceberg uses its own expression type. The `filters` parameter is applied as row-level pushdown only (pyarrow `table.filter()` after ArrowScan yields each batch).


After Iceberg resolves the surviving file list, the same `filters` expression SHALL
be passed to `StructuredDataset` for row-level pushdown within each file.

### Delegation `[v1]`
After file resolution, `IcebergDataset` SHALL construct a `StructuredDataset` and
delegate all iteration, split generation, and output conversion to it.
`IcebergDataset.create_dataloader()` SHALL return `(DataLoader, IcebergDataset)`.
The `IcebergDataset` instance SHALL expose `set_epoch()` by delegating to the inner dataset.

### Format `[v1]`
Format is always `"parquet"` — Iceberg data files are Parquet (or ORC).
`IcebergDataset` SHALL detect the format from the file extensions in the scan result.
When all files are `.parquet`, format is `"parquet"`. When `.orc`, format is `"orc"`.
Mixed formats within a single table are not supported in V1 — raise `ValueError`.

### Logging `[v1]`
SHALL log at `INFO`: catalog type, table name, snapshot id, file count, total size.
SHALL log at `DEBUG`: each resolved file path, partition, record_count.

---

## Interface

```python
class IcebergDataset:
    def __init__(
        self,
        table: str,
        catalog_config: dict,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
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
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        collate_fn: Callable | None = None,
    ) -> tuple[DataLoader, "IcebergDataset"]: ...
```

---

## Scenarios

#### Scenario: End-to-end Parquet read
- GIVEN a local Iceberg table with 3 Parquet files × 100 rows
- WHEN `IcebergDataset.create_dataloader(table=..., catalog_config=..., num_workers=0)` is called
- THEN total rows collected equals 300 and each batch is `dict[str, torch.Tensor]`

#### Scenario: Column projection
- GIVEN an Iceberg table with columns [feature_a, feature_b, label]
- WHEN `create_dataloader(..., columns=["feature_a", "label"])` is called
- THEN each batch contains only keys "feature_a" and "label"

#### Scenario: Predicate pushdown
- GIVEN an Iceberg table with feature_b values 0–99 across files
- WHEN `create_dataloader(..., filters=pc.field("feature_b") >= 50)` is called
- THEN only rows where feature_b >= 50 are returned

#### Scenario: Snapshot time travel
- GIVEN an Iceberg table with two snapshots
- WHEN `create_dataloader(..., snapshot_id=<old_id>)` is called
- THEN rows from the old snapshot are returned, not the current one

#### Scenario: TargetSizeSplitStrategy auto-selected
- GIVEN an Iceberg table where all files have record_count metadata
- WHEN `create_dataloader()` is called with default `split_strategy=None`
- THEN `TargetSizeSplitStrategy` is auto-selected (default strategy)

#### Scenario: Missing pyiceberg raises ImportError
- GIVEN `pyiceberg` is not installed
- WHEN `IcebergDataset.create_dataloader(...)` is called
- THEN `ImportError` is raised with `pip install torch-dataloader-utils[iceberg]`

#### Scenario: No rows dropped or duplicated
- GIVEN an Iceberg table with 3 files × 100 rows, row_id 0–299
- WHEN `create_dataloader` is fully iterated with `num_workers=0`
- THEN the set of all row_ids equals exactly {0..299} with no duplicates
