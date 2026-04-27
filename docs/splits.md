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

These workers are **not** DDP training ranks. In a multi-GPU setup, each DDP rank has its own `DataLoader` with its own `num_workers` I/O workers. V1 does not re-shard across DDP ranks automatically — see [Training Stack](training.md) for details.

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
| **ORC** | Whole file | No (V1) — stripes exist, planned for V2 | File size only (byte-based estimate) |
| **CSV** | Whole file | No | File size only |
| **JSON / JSONL** | Whole file | No | File size only |

**Parquet and Iceberg** are first-class: row group metadata is read once in the main process at split-generation time (no data scan, just the file footer — typically a few KB per file). Row groups are packed into `split_bytes`-sized chunks. A single large file can produce many chunks assigned to different workers.

**ORC, CSV, JSON, JSONL** are treated as unsplittable: each file becomes one chunk. The strategy still balances by estimated byte size across workers, but a single large file cannot be split between workers. For good parallelism with these formats, partition your data into many smaller files before training.

!!! tip "Large ORC or CSV files"
    If you have a few very large ORC or CSV files, sub-file splitting is not available. Split them into smaller files (e.g. 128–512 MiB each) to give the strategy enough chunks to balance across workers.

!!! note "V2 roadmap — ORC sub-file splitting"
    ORC files have **stripes** (equivalent to Parquet row groups) with row counts and byte offsets in the file footer. Sub-file splitting for ORC is technically feasible and is planned for V2. In V1, each ORC file is treated as a single unsplittable chunk.

### Sub-File Splitting

For large Parquet files, `TargetSizeSplitStrategy` generates multiple `Split` objects with disjoint `RowRange` values — a single file can be distributed across multiple workers.

```
Split(file=DataFileInfo, row_range=None)                    # whole file
Split(file=DataFileInfo, row_range=RowRange(0, 250_000))    # rows 0–250k
Split(file=DataFileInfo, row_range=RowRange(250_000, 250_000)) # rows 250k–500k
```

The reader uses `pq.ParquetFile.read_row_groups()` for true random access — only the assigned row groups are read.

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
- `shuffle=False` makes `set_epoch()` a no-op — splits are generated once and reused
- No record-level shuffle in V1 — chunk-level shuffle is sufficient for most training workloads

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
    ) -> list[Shard]:
        ...

loader, _ = StructuredDataset.create_dataloader(
    ..., split_strategy=MyStrategy()
)
```

No inheritance required — implement `generate()` and you're done.

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
