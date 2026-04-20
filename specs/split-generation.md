# Spec: Split Generation

## Core Principle

Split generation has three distinct concerns that must not be mixed:

```
File Discovery  →  Predicate Pushdown  →  Split Balancing  →  Split Assignment
(filesystem)       (pyarrow/pyiceberg)    (strategy)          (round-robin)
```

Predicate pushdown happens **before** splits — it reduces the file list. The split strategy only ever sees files that survived filtering.

---

## Data Classes

### `DataFileInfo` `[v1]`
Carries metadata for plain files (Parquet, ORC, CSV, JSON).

```
path: str                      # file URI
file_size: int | None          # bytes — from fsspec stat()
record_count: int | None       # rows — not available for plain files in V1
```

### `IcebergDataFileInfo(DataFileInfo)` `[v1]`
Extends `DataFileInfo` with Iceberg manifest metadata.

```
partition: dict[str, str] | None    # e.g. region=US, date=2024-01-01
snapshot_id: int | None             # for reproducibility and time travel
```

Note: `column_stats` are NOT stored here — column-level predicate pushdown is handled by `pyiceberg` during file discovery, before files reach the split layer.

### `RowRange` `[v1 — structure only, used in V2]`
Defines a row-level slice within a single file.

```
offset: int    # start row (inclusive)
length: int    # number of rows to read
```

### `FileSplit` `[v1]`
Pairs a file with an optional row range.

```
file: DataFileInfo
row_range: RowRange | None    # None = read entire file (V1)
                              # RowRange = sub-file slice (V2)
```

V1 always sets `row_range=None`. V2 strategies may populate `RowRange` to slice massive files across multiple workers without any breaking API changes.

### `Split` `[v1]`
A unit of work assigned to one DataLoader worker.

```
id: int
file_splits: list[FileSplit]    # files (and optional row ranges) this worker reads
row_count: int | None           # total rows across all file_splits (optional)
size_bytes: int | None          # total bytes across all file_splits (optional)
```

---

## Requirements

### `RoundRobinSplitStrategy` `[v1]`
The system SHALL distribute files across splits using round-robin assignment.
The system SHALL use this strategy when files are known to be equi-sized.
The system SHALL ignore `file_size` and `record_count` metadata.
The system SHALL produce exactly N splits where N equals `num_workers`.
No file SHALL appear in more than one split.

### `SizeBalancedSplitStrategy` `[v1]`
The system SHALL balance splits by total `record_count` when available (Iceberg).
The system SHALL fall back to `file_size` when `record_count` is None (plain files).
The system SHALL fall back to round-robin when neither is available.
The system SHALL use a greedy bin-packing algorithm — assign each file to the split with the current lowest total.

### Strategy Auto-Selection `[v1]`
The system SHALL auto-select the split strategy when `split_strategy=None`:
- If all files have `record_count` → `SizeBalancedSplitStrategy`
- Else if all files have `file_size` → `SizeBalancedSplitStrategy`
- Else → `RoundRobinSplitStrategy`
The system SHALL always allow the user to override via `split_strategy=`.
The system SHALL log the selected strategy at `INFO` level.

### `SplitStrategy` Protocol `[v1]`
The system SHALL define `SplitStrategy` as a `Protocol` — not an ABC.
Any class with a matching `generate()` method SHALL satisfy the protocol.
No inheritance SHALL be required from user-defined strategies.

```python
class SplitStrategy(Protocol):
    def generate(self, files: list[DataFileInfo], num_workers: int, epoch: int) -> list[Split]:
        ...
```

### Shuffle `[v1]`
The system SHALL shuffle the file list before split assignment when `shuffle=True`.
The system SHALL use `shuffle_seed + epoch` as the random seed.
The system SHALL only regenerate splits when shuffle is enabled — reuse cached splits otherwise.
The system SHALL NOT mutate the input file list.

### `num_workers` Auto-Detection `[v1]`
The system SHALL accept `num_workers=None` to trigger auto-detection.
Auto-detection SHALL use `max(1, os.cpu_count() - 1)`.
The system SHALL log the resolved value at `INFO` level.
The system SHALL treat `num_workers=0` as single-process mode (PyTorch convention).

### Split Timing `[v1]`
File discovery SHALL happen once at `create_dataloader()` time.
Split generation SHALL happen at the start of each `__iter__()` call.
Split generation SHALL be skipped when `shuffle=False` and splits are already cached.

### Sub-file Row Range Splitting `[v2]`
The system SHALL allow a single file to be split across multiple workers using `RowRange`.
The system SHALL compute `RowRange` offsets using `record_count` from `DataFileInfo`.
Sub-file splitting SHALL only apply to Parquet and Iceberg formats (not CSV/JSON).

---

## Scenarios

#### Scenario: Round-robin even distribution
- GIVEN 8 equi-sized files and 4 workers
- WHEN `RoundRobinSplitStrategy` generates splits
- THEN each split contains exactly 2 files

#### Scenario: Round-robin uneven distribution
- GIVEN 9 files and 4 workers
- WHEN `RoundRobinSplitStrategy` generates splits
- THEN no file appears in more than one split
- AND the largest split has at most one more file than the smallest

#### Scenario: Size-balanced with record_count (Iceberg)
- GIVEN 4 IcebergDataFileInfo files with record_counts [1000, 500, 300, 200] and 2 workers
- WHEN `SizeBalancedSplitStrategy` generates splits
- THEN split totals are as equal as possible (e.g. [1000] vs [500+300+200=1000])

#### Scenario: Size-balanced with file_size (plain files)
- GIVEN 4 DataFileInfo files with file_sizes [100, 50, 30, 20] and 2 workers
- WHEN `SizeBalancedSplitStrategy` generates splits
- THEN split totals are as equal as possible by bytes

#### Scenario: Size-balanced fallback to round-robin
- GIVEN files with no file_size and no record_count
- WHEN `SizeBalancedSplitStrategy` generates splits
- THEN falls back to round-robin assignment

#### Scenario: Auto-selection — Iceberg files
- GIVEN all files are IcebergDataFileInfo with record_count populated
- WHEN split_strategy=None
- THEN SizeBalancedSplitStrategy is selected automatically

#### Scenario: Auto-selection — plain files with size
- GIVEN all files are DataFileInfo with file_size populated
- WHEN split_strategy=None
- THEN SizeBalancedSplitStrategy is selected automatically

#### Scenario: Auto-selection — plain files without size
- GIVEN files with no size metadata
- WHEN split_strategy=None
- THEN RoundRobinSplitStrategy is selected automatically

#### Scenario: Shuffle reproducibility
- GIVEN shuffle=True and shuffle_seed=42
- WHEN two split generations run with the same epoch
- THEN both produce identical split assignments

#### Scenario: Epoch variation
- GIVEN shuffle=True and shuffle_seed=42
- WHEN splits are generated for epoch 0 and epoch 1
- THEN the file order differs between epochs

#### Scenario: No shuffle — splits cached
- GIVEN shuffle=False
- WHEN __iter__() is called for epoch 0 and epoch 1
- THEN split generation runs only once

#### Scenario: Custom strategy — no inheritance needed
- GIVEN a user-defined class with a generate() method
- WHEN passed as split_strategy=
- THEN it is used without any inheritance from SplitStrategy

#### Scenario: V1 row_range is always None
- GIVEN any V1 strategy (RoundRobin or SizeBalanced)
- WHEN splits are generated
- THEN every FileSplit has row_range=None

#### Scenario: V2 sub-file splitting for massive file `[v2]`
- GIVEN 1 file with record_count=1_000_000 and 4 workers
- WHEN a sub-file strategy generates splits
- THEN each split contains one FileSplit with RowRange(offset=N*250_000, length=250_000)
- AND no rows overlap across splits
