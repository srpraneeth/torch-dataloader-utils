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
- `path: str` — passed to `discover_files()`
- `format: str` — parquet, orc, csv, json, jsonl
- `batch_size: int` — default 1024
- `columns: list[str] | None` — column projection
- `filters: pc.Expression | None` — predicate pushdown
- `shuffle: bool` — default False
- `shuffle_seed: int` — default 42
- `split_strategy: SplitStrategy | None` — None = auto-select
- `num_workers: int | None` — None = auto-detect via `max(1, os.cpu_count() - 1)`
- `output_format: str` — torch | numpy | arrow | dict, default "torch"
- `storage_options: dict | None` — forwarded to fsspec and read_split
- `collate_fn: Callable | None` — forwarded to DataLoader

The system SHALL raise `ValueError` for unsupported `format` values at construction time.
The system SHALL raise `ValueError` for unsupported `output_format` values at construction time.
The system SHALL raise `ValueError` when `output_format` is `"arrow"` or `"dict"` and `collate_fn` is None.

### File Discovery `[v1]`
File discovery SHALL happen once at `create_dataloader()` time via `discover_files()`.
The discovered `list[DataFileInfo]` SHALL be stored on the dataset instance.

### Split Generation `[v1]`
Split generation SHALL happen at the start of each `__iter__()` call.
Split generation SHALL be skipped when `shuffle=False` and splits are already cached.
When `shuffle=True` splits SHALL be regenerated each epoch using `shuffle_seed + epoch`.
The `_epoch` counter SHALL increment each time `__iter__()` is called.

### Strategy Auto-Selection `[v1]`
When `split_strategy=None` the system SHALL auto-select:
- All files have `record_count` → `SizeBalancedSplitStrategy`
- All files have `file_size` → `SizeBalancedSplitStrategy`
- Otherwise → `RoundRobinSplitStrategy`
The system SHALL log the selected strategy at `INFO` level.

### Worker Assignment `[v1]`
Each worker SHALL read exactly one `Split`.
The system SHALL use `torch.utils.data.get_worker_info()` to determine the current worker index.
When `num_workers=0` (single process) the system SHALL use split index 0.
The system SHALL raise `RuntimeError` if `get_worker_info().num_workers` does not match the number of splits.

### Iteration `[v1]`
Each worker SHALL call `read_split()` with its assigned `Split`.
Each worker SHALL yield one item per `RecordBatch` after output format conversion.
The system SHALL NOT buffer or accumulate batches — yield immediately.

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
It SHALL perform file discovery, strategy selection, and return a `DataLoader`.
It SHALL construct `DataLoader(dataset, batch_size=None, num_workers=N, collate_fn=collate_fn)`.
`batch_size=None` is always passed — Arrow owns batching, not DataLoader.

### Logging `[v1]`
The system SHALL log at `INFO`: path, format, num_workers, split strategy selected, file count.
The system SHALL log at `DEBUG`: worker index, split id assigned, files in split.

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
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
    ) -> None: ...

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
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
    ) -> DataLoader: ...
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
- WHEN `__iter__()` is called for epoch 0 and epoch 1
- THEN the file order differs between epochs

#### Scenario: No shuffle — splits cached
- GIVEN `shuffle=False`
- WHEN `__iter__()` is called twice
- THEN split generation runs only once

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
