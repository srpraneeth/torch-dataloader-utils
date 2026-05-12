# Splits & Workers

## What Is a Worker Here?

**`num_workers` refers to `DataLoader` I/O workers — not training processes (DDP ranks).**

PyTorch's `DataLoader` spawns background processes to prefetch data while the GPU trains. Each of these is an I/O worker: it reads files, decodes batches, and places them in a shared queue that the training process consumes.

```
Training process (GPU)
    ↑ consumes batches from shared queue
    │
    ├── DataLoader Worker 0  ← reads its assigned files, decodes, batches
    ├── DataLoader Worker 1  ← reads its assigned files, decodes, batches
    ├── DataLoader Worker 2  ← reads its assigned files, decodes, batches
    └── DataLoader Worker 3  ← reads its assigned files, decodes, batches
```

These workers are **not** DDP training ranks. In a multi-GPU setup, each DDP rank has its own `DataLoader` with its own `num_workers` I/O workers. Use `num_ranks` and `rank` on `create_dataloader()` for rank-level file sharding — see [Rank-Aware DDP Sharding](#rank-aware-ddp-sharding) below.

## How Splits Work

Files are pre-partitioned into `Shard` objects — one per `DataLoader` worker — before iteration begins. This happens once at `create_dataloader()` time.

```
File Discovery → Scan-Level Pruning → Split Generation → Shard Assignment
(fsspec/pyiceberg)  (scan_filter)      (strategy)         (to DataLoader workers)
```

Each worker owns exactly one shard and reads its assigned files sequentially. Workers never communicate or share data.

## Split Strategies

| Strategy | When | Balances by |
|----------|------|-------------|
| `TargetSizeSplitStrategy` | Default (non-empty file list) | Row count via LPT scheduling |
| `RoundRobinSplitStrategy` | Fallback (empty file list) | File count |

### TargetSizeSplitStrategy

Chunks are assigned to workers using **greedy min-heap (LPT scheduling)** — always assigns the next chunk to the least-loaded worker. Near-perfectly balanced row counts even for very unequal file sizes.

```python
# Tune chunk size
loader, _ = StructuredDataset.create_dataloader(
    ...,
    split_bytes="64MiB",    # string or int bytes
    # or
    split_rows=50_000,      # overrides split_bytes
)
```

#### Behavior by format

| Format | Chunk granularity | Sub-file splitting | How row count is known |
|--------|-------------------|--------------------|------------------------|
| **Parquet** | Row group | Yes | Footer metadata (no data read) |
| **Iceberg** | Row group (resolves to Parquet files) | Yes | Footer metadata via pyiceberg scan |
| **ORC** | Stripe (uniform approximation) | Yes — stripe-boundary chunks | Approximate (nrows/nstripes) |
| **CSV** | Whole file | No | File size only |
| **JSON / JSONL** | Whole file | No | File size only |

**Parquet and Iceberg** are first-class: row group metadata is read once in the main process at split-generation time (no data scan, just the file footer — typically a few KB per file). Row groups are packed into `split_bytes`-sized chunks. A single large file can produce many chunks assigned to different workers.

**ORC** files are split at stripe boundaries. Because PyArrow does not expose per-stripe row counts in the footer, the row count is approximated uniformly as `nrows / nstripes`. The reader calls `ORCFile.read_stripe()` for each assigned stripe — true random access, no full scan.

**CSV, JSON, JSONL** are treated as unsplittable: each file becomes one chunk. For good parallelism with these formats, partition your data into many smaller files before training.

!!! tip "Large CSV or JSONL files"
    If you have a few very large CSV or JSONL files, sub-file splitting is not available. Split them into smaller files (e.g. 128–512 MiB each) to give the strategy enough chunks to balance across workers.

### Sub-File Splitting

For large Parquet and ORC files, `TargetSizeSplitStrategy` generates multiple `Split` objects with disjoint `RowRange` values — a single file can be distributed across multiple workers.

```
Split(file=DataFileInfo, row_range=None)                    # whole file
Split(file=DataFileInfo, row_range=RowRange(0, 250_000))    # rows 0–250k
Split(file=DataFileInfo, row_range=RowRange(250_000, 250_000)) # rows 250k–500k
```

For **Parquet**, the reader uses `pq.ParquetFile.read_row_groups()` — only the assigned row groups are read (true random access, row-group granularity).

For **ORC**, the reader uses `ORCFile.read_stripe()` — only the assigned stripes are read (stripe granularity). Row counts are approximated uniformly since PyArrow does not expose per-stripe row counts.

## Shuffle

Shuffle operates at the chunk level before shard assignment:

```python
loader, dataset = StructuredDataset.create_dataloader(
    ..., shuffle=True, shuffle_seed=42
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)  # seed = shuffle_seed + epoch
    for batch in loader:
        ...
```

- Call `set_epoch()` in the **main process** before each epoch
- With `shuffle=False`, `set_epoch()` can be omitted — splits are always generated in the same deterministic order regardless of epoch number
- No record-level shuffle — chunk-level shuffle is sufficient for most training workloads

## Rank-Aware DDP Sharding

`num_ranks` and `rank` add a second level of partitioning above the existing worker-level splits. The hierarchy is:

```
Rank partitioning  →  Worker partitioning (within each rank)
(num_ranks / rank)     (num_workers / split strategy)
```

All splits are generated first, then interleaved by rank:

```
rank_splits = all_splits[rank::num_ranks]
```

This means rank 0 gets splits 0, 2, 4, …; rank 1 gets splits 1, 3, 5, …; and so on. Each rank then further divides its own `rank_splits` across its `num_workers` I/O workers.

### Usage

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=dist.get_world_size(),   # total DDP ranks
    rank=dist.get_rank(),              # this process's rank
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

### With Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=accelerator.num_processes,
    rank=accelerator.process_index,
)

loader = accelerator.prepare(loader)   # optional — adds gradient sync wrappers
```

### With Horovod

```python
import horovod.torch as hvd

hvd.init()

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    num_ranks=hvd.size(),
    rank=hvd.rank(),
)
```

!!! note "Default behaviour (V1 compatible)"
    `num_ranks=1, rank=0` are the defaults — all splits go to the single rank, identical to V1 behaviour.

!!! note "Global rank vs local rank"
    Use the **global** rank and world size (across all nodes), not the per-node local rank. In PyTorch DDP: `dist.get_rank()` is global. In Accelerate: `accelerator.process_index` is global. Local rank (`LOCAL_RANK` env var or `accelerator.local_process_index`) is the GPU index within a single node and should not be used here.

## num_workers

`num_workers` controls how many **DataLoader I/O background processes** spawn to prefetch data. It has nothing to do with GPU count or DDP rank count.

```python
num_workers=None   # auto-detect: max(1, os.cpu_count() - 1), logged at INFO
num_workers=0      # single process — no forking, useful for debugging
num_workers=4      # 4 background I/O processes
```

A good starting point is `num_workers=4`. For cloud storage (S3, GCS), more workers overlap network latency well — try 8–16. For local NVMe, 2–4 is usually enough before CPU becomes the bottleneck.

!!! tip "Debugging"
    Use `num_workers=0` when debugging — all reads happen in the main process, making stack traces and breakpoints work normally.

## Custom Strategy

```python
from torch_dataloader_utils.splits.core import DataFileInfo, Shard

class MyStrategy:
    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int,
        num_ranks: int = 1,   # optional — V1 strategies without these params still work
        rank: int = 0,
    ) -> list[Shard]:
        ...

loader, _ = StructuredDataset.create_dataloader(
    ..., split_strategy=MyStrategy()
)
```

No inheritance required — implement `generate()` and you're done. V1 strategies that do not accept `num_ranks` / `rank` continue to work — the library inspects the signature and omits those arguments if not present.

## File Metadata

The split layer operates on two file metadata types:

```python
DataFileInfo          # plain files: path, file_size, record_count
IcebergDataFileInfo   # extends DataFileInfo: + partition, snapshot_id
```

Both satisfy the same interface — `SplitStrategy.generate()` accepts `list[DataFileInfo]` and works with either. Iceberg's richer metadata (partition values, column-level statistics from the snapshot manifest) is used by `IcebergDataset` during the scan phase for predicate pushdown and file pruning — the split strategy itself only sees `path`, `file_size`, and `record_count` from the base class.

Custom strategies can `isinstance`-check for `IcebergDataFileInfo` if they want partition-aware assignment logic:

```python
from torch_dataloader_utils.splits.core import DataFileInfo, IcebergDataFileInfo, Shard

class PartitionAwareStrategy:
    def generate(self, files: list[DataFileInfo], num_workers: int, epoch: int) -> list[Shard]:
        for f in files:
            if isinstance(f, IcebergDataFileInfo):
                partition = f.partition   # e.g. {"region": "us-east-1"}
            ...
```
