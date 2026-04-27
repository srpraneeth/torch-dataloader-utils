# StructuredDataset

`StructuredDataset` reads file-based formats (Parquet, ORC, CSV, JSON, JSONL) from any `fsspec`-compatible filesystem.

## create_dataloader()

```python
from torch_dataloader_utils import StructuredDataset
import pyarrow.compute as pc

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://my-bucket/data/train/",
    format="parquet",                         # parquet | orc | csv | json | jsonl
    num_workers=None,                         # None = auto-detect, 0 = single process
    batch_size=1024,
    columns=["feature_a", "feature_b", "label"],
    filters=pc.field("date") > "2024-01-01",  # predicate pushdown via pyarrow
    shuffle=True,
    shuffle_seed=42,
    split_bytes="128MiB",                     # target chunk size (default 128 MiB)
    split_rows=None,                          # target rows per chunk (overrides split_bytes)
    split_strategy=None,                      # None = auto-select TargetSizeSplitStrategy
    output_format="torch",                    # torch | numpy | arrow | dict
    storage_options={"key": "...", "secret": "..."},
    partitioning="hive",                      # None | "hive"
)
```

Returns `(DataLoader, dataset)`. Keep a reference to `dataset` to call `set_epoch()` each epoch when `shuffle=True`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Directory, glob, or single file path |
| `format` | `str` | — | `parquet`, `orc`, `csv`, `json`, `jsonl` |
| `num_workers` | `int \| None` | `None` | DataLoader workers. `None` = `max(1, cpu_count - 1)`. `0` = single process |
| `batch_size` | `int` | `1024` | Rows per batch |
| `columns` | `list[str] \| None` | `None` | Column projection. `None` = all columns |
| `filters` | `pc.Expression \| None` | `None` | Row-level predicate pushdown via pyarrow |
| `shuffle` | `bool` | `False` | Shuffle chunks before assigning to workers |
| `shuffle_seed` | `int` | `42` | Base seed — actual seed is `shuffle_seed + epoch` |
| `split_bytes` | `int \| str \| None` | `None` | Target bytes per chunk. Strings like `"128MiB"` accepted. `None` = 128 MiB |
| `split_rows` | `int \| None` | `None` | Target rows per chunk. Overrides `split_bytes` |
| `split_strategy` | `SplitStrategy \| None` | `None` | Custom strategy. `None` = auto-select |
| `output_format` | `str` | `"torch"` | `torch`, `numpy`, `arrow`, `dict` |
| `storage_options` | `dict \| None` | `None` | Forwarded to fsspec (credentials, endpoint, etc.) |
| `collate_fn` | `Callable \| None` | `None` | Custom DataLoader collate function |
| `partitioning` | `str \| None` | `None` | `"hive"` decodes `key=value` directory segments as columns |

## Epoch Reshuffling

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,
    num_workers=4,
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)    # regenerates splits with seed + epoch
    for batch in loader:
        ...
```

Call `set_epoch()` in the **main process** before each epoch, not inside a worker.

## Custom Split Strategy

```python
from torch_dataloader_utils.splits.core import DataFileInfo, Shard

class MyStrategy:
    def generate(self, files: list[DataFileInfo], num_workers: int, epoch: int) -> list[Shard]:
        ...

loader, _ = StructuredDataset.create_dataloader(
    ..., split_strategy=MyStrategy()
)
```

No inheritance required — just implement `generate()`.

## Collate Function

`create_dataloader()` handles collation automatically based on `output_format`:

| `output_format` | `collate_fn` required? | Default behaviour |
|-----------------|------------------------|-------------------|
| `"torch"` | No | PyTorch default collate (stacks tensors) |
| `"numpy"` | No | Auto-generated passthrough (`lambda x: x`) |
| `"arrow"` | **Yes** | Raises `ValueError` if not provided |
| `"dict"` | **Yes** | Raises `ValueError` if not provided |

```python
# arrow / dict — must provide collate_fn
loader, _ = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    output_format="arrow",
    collate_fn=lambda x: x,
)
```

!!! note "Direct constructor"
    The `ValueError` for missing `collate_fn` is raised at **constructor** time when using `StructuredDataset(...)` directly. `create_dataloader()` auto-generates a passthrough for `"numpy"` so you never need to pass one for numpy mode.

## Advanced: Direct Constructor

```python
from torch.utils.data import DataLoader
from torch_dataloader_utils import StructuredDataset

ds = StructuredDataset(files=..., format="parquet", ...)
loader = DataLoader(ds, batch_size=None, num_workers=4)
loader = accelerator.prepare(loader)
```
