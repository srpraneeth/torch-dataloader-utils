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
    num_ranks=1,                              # total DDP world size (default 1 = single process)
    rank=0,                                   # this process's global DDP rank (default 0)
    show_progress=False,                      # tqdm progress bars per worker per file
    progress_interval_sec=120,                # how often to refresh bars (seconds)
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
| `shuffle_buffer_size` | `int \| None` | `None` | Record-level shuffle buffer size (rows). `None` = no record shuffle |
| `split_bytes` | `int \| str \| None` | `None` | Target bytes per chunk. Strings like `"128MiB"` accepted. `None` = 128 MiB |
| `split_rows` | `int \| None` | `None` | Target rows per chunk. Overrides `split_bytes` |
| `split_strategy` | `SplitStrategy \| None` | `None` | Custom strategy. `None` = auto-select |
| `output_format` | `str` | `"torch"` | `torch`, `numpy`, `arrow`, `dict` |
| `storage_options` | `dict \| None` | `None` | Forwarded to fsspec (credentials, endpoint, etc.) |
| `collate_fn` | `Callable \| None` | `None` | Custom DataLoader collate function |
| `partitioning` | `str \| None` | `None` | `"hive"` decodes `key=value` directory segments as columns |
| `num_ranks` | `int` | `1` | Total DDP world size. Default `1` = single-process (V1 behaviour) |
| `rank` | `int` | `0` | This process's global DDP rank (0-indexed). Default `0` |
| `show_progress` | `bool` | `False` | Show tqdm progress bars (one per worker per file). Requires `tqdm` |
| `progress_interval_sec` | `float` | `120.0` | How often to refresh progress bars (seconds) |

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

## Record-Level Shuffle

`shuffle=True` randomises chunk order — but within a chunk, rows are read in file order. If your Parquet files are sorted by timestamp or user ID, consecutive batches will still be correlated.

`shuffle_buffer_size` adds a per-worker reservoir buffer that mixes rows across chunks:

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,               # chunk-level: reorders file chunks across epochs
    shuffle_buffer_size=50_000, # record-level: mixes rows within each worker's stream
    num_workers=4,
)
```

The two levels are independent — use either or both:

| `shuffle` | `shuffle_buffer_size` | Effect |
|-----------|----------------------|--------|
| `True` | `None` | Chunk order randomised, rows in file order |
| `False` | `50_000` | Fixed chunk order, rows mixed within each worker |
| `True` | `50_000` | Chunk order randomised + rows mixed (recommended for training) |

**Buffer sizing:**

| `shuffle_buffer_size` | 20 float32 cols | 100 float32 cols |
|-----------------------|-----------------|------------------|
| 10,000 rows | 0.8 MB / worker | 4 MB / worker |
| 50,000 rows | 4 MB / worker | 20 MB / worker |
| 100,000 rows | 8 MB / worker | 40 MB / worker |

With `num_workers=8` and `shuffle_buffer_size=50_000` (100 cols): 8 × 20 MB = 160 MB total. For full shuffle quality set `shuffle_buffer_size` to `dataset_size / num_workers`.

!!! note "Memory lives in worker processes"
    Each worker maintains its own independent buffer. No IPC occurs until a completed output batch crosses the DataLoader pipe — the buffer is entirely local to the worker heap.



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

## Distributed Training (DDP)

Pass `num_ranks` and `rank` to shard files across DDP processes. Each rank receives an interleaved subset of all splits — rank 0 gets splits 0, 2, 4, …; rank 1 gets splits 1, 3, 5, …; and so on.

### PyTorch DDP

```python
import torch.distributed as dist
from torch_dataloader_utils import StructuredDataset

dist.init_process_group(backend="nccl")

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=dist.get_world_size(),
    rank=dist.get_rank(),
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

### HuggingFace Accelerate

```python
from accelerate import Accelerator
from torch_dataloader_utils import StructuredDataset

accelerator = Accelerator()

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=accelerator.num_processes,
    rank=accelerator.process_index,     # global rank, not local_process_index
)

loader = accelerator.prepare(loader)   # optional — adds gradient sync wrappers

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

!!! note "Global rank vs local rank"
    Use `accelerator.process_index` (global rank across all nodes), not `accelerator.local_process_index` (GPU index within one node). The same applies to PyTorch DDP: `dist.get_rank()` is the correct global rank.

## Advanced: Direct Constructor

```python
from torch.utils.data import DataLoader
from torch_dataloader_utils import StructuredDataset

ds = StructuredDataset(files=..., format="parquet", ...)
loader = DataLoader(ds, batch_size=None, num_workers=4)
loader = accelerator.prepare(loader)
```

---

## Observability

`StructuredDataset` emits structured logs at every stage — startup summary, split assignment table, load balance warnings, per-file logs, progress bars, and epoch summaries.

See **[Observability](observability.md)** for the full reference.
