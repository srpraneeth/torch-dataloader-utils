# torch-dataloader-utils

A lightweight PyTorch library for reading structured tabular data from cloud object storage directly into a `DataLoader`. No Ray, no Spark, no heavy infrastructure.

---

## Problem

PyTorch has no built-in story for reading structured data from cloud storage into a `DataLoader`. The existing ecosystem either pulls in heavy dependencies, lacks Iceberg support, or is abandoned.

| Solution | Problem |
|----------|---------|
| `torchdata` | Solves checkpointing and pipeline composition; no built-in cloud file sharding or Iceberg support |
| Ray Data | Requires a Ray cluster — overkill for most |
| HuggingFace `datasets` | No Iceberg, opinionated about data structure |
| `tf.data` + `tfio` | TensorFlow only, no Iceberg |
| WebDataset | Designed for unstructured data (images, audio) |

Most teams training with DDP or FSDP on data stored in S3/GCS/Azure have no lightweight option to get Parquet or Iceberg tables into a `DataLoader` without pulling in a distributed compute engine.

---

## Solution

A thin library with three responsibilities:

1. **Discover** files from any filesystem (S3, GCS, Azure, local, MinIO) using `fsspec`
2. **Read** structured formats (Parquet, ORC, CSV, JSON, Iceberg) using `pyarrow` and `pyiceberg`
3. **Distribute** data across DataLoader workers using a pre-computed split system

Returns a standard `torch.utils.data.DataLoader` — no changes needed anywhere else in the training stack.

---

## Architecture

```
FilesystemLayer   →   FormatLayer          →   DatasetLayer
fsspec                pyarrow.dataset           torch.IterableDataset
                      pyiceberg (Iceberg only)
```

### Filesystem Layer
`fsspec` provides a uniform interface across all storage backends. Credentials and endpoint configuration are passed via `storage_options` and forwarded directly to fsspec — the library adds no credential management of its own.

### Format Layer
`pyarrow.dataset` handles Parquet, ORC, CSV, and JSON with column projection and predicate pushdown built in. For Iceberg, `pyiceberg` connects to the catalog, resolves the table to a list of data file URIs, and passes them to `pyarrow.dataset`. Iceberg is a table format on top of Parquet/ORC — the actual reading is identical after resolution.

### Dataset Layer
Extends `torch.utils.data.IterableDataset`. Files are pre-partitioned into `Split` objects — one per worker — before the DataLoader starts. Each worker owns exactly one split and reads its files sequentially. `DataLoader` is always constructed with `batch_size=None` because Arrow owns batching internally.

---

## Architecture Goals

- **Lightweight** — no Ray, no Spark, no distributed compute engine required
- **Filesystem-agnostic** — any fsspec-compatible backend works out of the box
- **Format-agnostic** — same API regardless of file format or table format
- **PyTorch-native** — returns a standard `DataLoader`, composes with DDP, FSDP, and Accelerate without modification
- **Arrow-native batching** — `pyarrow` reads in `RecordBatch` chunks natively; no redundant re-batching at the DataLoader level
- **Predictable worker distribution** — splits are pre-computed, not derived at runtime, so each worker's data assignment is deterministic and inspectable

---

## Split-Based Worker Distribution

Files are pre-partitioned into `Split` objects — one per worker — before iteration begins. File discovery happens once at `create_dataloader()`. Split generation happens at the start of each epoch, reused as-is when shuffle is off.

The pipeline is:

```
File Discovery → Scan-Level Pruning → Split Generation → Split Assignment
(fsspec/pyiceberg)  (scan_filter)      (strategy)         (to workers)
```

Scan-level predicate pushdown reduces the file list **before** splits are generated — the split strategy only sees files that survived filtering.

### Split Strategies

Two built-in strategies, both satisfying the `SplitStrategy` protocol:

| Strategy | When to use | Balances by |
|----------|-------------|-------------|
| `RoundRobinSplitStrategy` | Empty file list fallback | File count |
| `TargetSizeSplitStrategy` | Default for non-empty file lists | Row count (LPT scheduling) |

`TargetSizeSplitStrategy` behaviour:
- **Parquet files**: reads row group metadata from the file footer (no data scan). Packs consecutive row groups into chunks targeting `split_bytes` (default 128 MiB). Each chunk is a `FileSplit` with a `RowRange` specifying the exact row range.
- **Non-Parquet files** (ORC, CSV, JSONL): each file is treated as a single unsplittable chunk — no footer metadata available.
- `split_rows` overrides `split_bytes` when both are set — produces row-count-balanced splits.
- Chunks are distributed across workers using a greedy min-heap (LPT scheduling) — sorts largest chunks first, assigns each to the least-loaded worker. Minimises maximum worker load for unequal chunk sizes.

`create_dataloader()` auto-selects:
- Non-empty file list → `TargetSizeSplitStrategy`
- Empty file list → `RoundRobinSplitStrategy`

Custom strategies require no inheritance — just a `generate()` method:

```python
class MyStrategy:
    def generate(self, files, num_workers, epoch) -> list[Split]:
        ...

loader = StructuredDataset.create_dataloader(..., split_strategy=MyStrategy())
```

### File Metadata

```
DataFileInfo              — plain files: path, file_size, record_count
IcebergDataFileInfo       — extends DataFileInfo with: partition, snapshot_id
```

Iceberg column-level statistics are used for predicate pushdown during file discovery — they do not flow into the split layer.

### Sub-file Splitting — `FileSplit` and `RowRange`

Each `Split` holds a list of `FileSplit` objects — a file paired with an optional row range:

```
FileSplit(file=DataFileInfo, row_range=None)              # whole file
FileSplit(file=DataFileInfo, row_range=RowRange(0, 250k)) # sub-file row slice
```

For Parquet files with multiple row groups, `TargetSizeSplitStrategy` generates multiple `FileSplit` objects with disjoint `RowRange` values for a single file — enabling a large file to be distributed across multiple workers. The reader uses `pq.ParquetFile.read_row_groups()` to seek directly to the assigned row groups (true random access, no full scan).

### Shuffle

Shuffle operates at the chunk list level before split assignment — no record-level shuffle. Each epoch uses `seed + epoch` for reproducibility.

### `num_workers`

```python
num_workers=None   # auto-detect: max(1, os.cpu_count() - 1), logged at INFO
num_workers=0      # single process — useful for debugging
num_workers=4      # explicit
```

---

## API

### File-Based Formats

```python
from torch_dataloader_utils import StructuredDataset

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
    partitioning="hive",                      # None | "hive" — decode key=value directory segments
)
```

### Iceberg Tables

```python
from torch_dataloader_utils import IcebergDataset
import pyarrow.compute as pc

loader, dataset = IcebergDataset.create_dataloader(
    table="my_catalog.my_db.my_table",
    catalog_config={
        "type": "rest",                       # rest | glue | hive | jdbc
        "uri": "https://catalog.example.com",
        "credential": "token:abc123",
    },
    num_workers=4,
    batch_size=1024,
    columns=["feature_a", "feature_b", "label"],
    filters=pc.field("region_id") >= 5,       # auto-prunes files AND filters rows
    snapshot_id=8271638172635,                # optional — time travel
    shuffle=True,
    split_bytes="64MiB",                      # target chunk size
    output_format="torch",
)
```

Passing `filters` is all you need. The library auto-translates common pyarrow expressions (`>=`, `>`, `<=`, `<`, `==`, `!=`, `&`, `|`) into a native pyiceberg expression and pushes it into `table.scan(row_filter=...)` at `plan_files()` time — pruning entire partitions and files **before** splits are generated. The same expression is also applied row-level inside workers. You can verify this in logs:

```
INFO  Auto-derived scan_filter:  pc.Expression (region_id >= 5)  →  pyiceberg GreaterThanOrEqual(...)
INFO  Iceberg scan complete: files=4  (down from 6 without filter)
```

For advanced cases where you need explicit control over the two layers separately:

```python
from pyiceberg.expressions import GreaterThan

loader, dataset = IcebergDataset.create_dataloader(
    ...
    filters=pc.field("score") > 0.9,         # row-level filter (applied in workers)
    scan_filter=GreaterThan("partition_dt", 20240101),  # file/partition pruning only
)
```

When `scan_filter` is provided explicitly, auto-derivation is skipped entirely.

### Collate Function

By default the PyTorch collate function is used, which works correctly with `output_format="torch"`. Pass a custom collate function for variable-length sequences, padding, or custom stacking logic:

```python
loader, _ = StructuredDataset.create_dataloader(
    path="s3://my-bucket/data/train/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    collate_fn=my_collate,    # optional, defaults to None → PyTorch default
)
```

> **Note:** If `output_format` is `"arrow"` or `"dict"` and no `collate_fn` is provided, the library raises a clear error at construction time rather than letting PyTorch fail silently during iteration.

### Advanced Usage (Escape Hatch)

```python
# Direct constructor for custom DataLoader setup or Accelerate workflows
ds = StructuredDataset(path="...", format="parquet", ...)
loader = DataLoader(ds, batch_size=None, num_workers=4)
loader = accelerator.prepare(loader)
```

### Output Formats

| `output_format` | Type returned per batch |
|-----------------|------------------------|
| `"torch"` | `dict[str, torch.Tensor]` — default, model-ready |
| `"numpy"` | `dict[str, np.ndarray]` — lighter weight |
| `"arrow"` | `pyarrow.RecordBatch` — no conversion overhead |
| `"dict"` | `dict[str, list]` — plain Python |

---

## Integration with Training Stack

```
[S3 / GCS / Azure / Iceberg]
           ↓
[StructuredDataset / IcebergDataset]    ← this library
           ↓
[DataLoader(batch_size=None)]
           ↓
[accelerator.prepare(loader)]           ← Accelerate shards across DDP ranks
           ↓
[DDP / FSDP training loop]
```

No modifications needed at any other layer. `create_dataloader()` returns a standard `DataLoader` — the rest of the stack is unaware this library exists.

### How This Differs from Accelerate / Horovod / DeepSpeed Sharding

Frameworks like Accelerate, Horovod, and DeepSpeed coordinate distributed training by assigning each process a global rank. For **map-style datasets** (with known length and indexing), PyTorch's `DistributedSampler` automatically splits indices by rank — each process sees a unique shard.

For **iterable datasets** (unknown length, streaming), no automatic sharding is provided. These frameworks still load all data into each worker and apply filtering at the worker level — the entire dataset is read before any subset is selected. This is wasteful for large datasets on cloud storage.

**This library takes a different approach — pre-computed splits with predicate pushdown:**

```
Standard iterable dataset approach:
  Each worker reads ALL files → filters to its shard → discards the rest
  Cost: full dataset read on every worker

This library's approach:
  Splits computed in main process → each worker assigned only its files
  Each worker reads ONLY its assigned files — no filtering, no wasted I/O
  Cost: each file read exactly once, by exactly one worker
```

The split assignment happens at `create_dataloader()` time — before any worker starts. Each worker is only ever aware of its own `Split` object and has no visibility into other workers' data. This makes the pipeline efficient for cloud storage where list and read operations have real latency and cost.

### Epoch Reshuffling with set_epoch()

For shuffle support across epochs with multi-worker DataLoaders, call `set_epoch(n)` on the dataset in the main process before each epoch — the same pattern as PyTorch's `DistributedSampler`:

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,
    shuffle_seed=42,
    num_workers=4,
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)       # regenerates splits with new shuffle order
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

`set_epoch()` must be called in the main process — not inside a worker. Workers receive the updated splits via pickling when the DataLoader starts the next epoch.

When `shuffle=False`, `set_epoch()` is a no-op — splits are generated once and reused.

### V2: Accelerate-Native Integration

In V2, `create_dataloader()` will accept an `accelerator` parameter and handle DDP rank-aware split assignment automatically — each DDP rank gets a disjoint set of splits, with no overlap across ranks:

```python
# V2 — planned
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    accelerator=accelerator,    # rank-aware split assignment
)
```

In V1, pass the loader to `accelerator.prepare()` after construction — Accelerate wraps the DataLoader but does not re-shard the underlying splits, so each rank reads all data. For true rank-level sharding in V1, construct separate datasets per rank using the `split_strategy` escape hatch.

---

## Dependencies

### Core

| Library | Version | Role |
|---------|---------|------|
| Python | >=3.11 | Type hint syntax, `tomllib` built-in |
| PyTorch | >=2.2 | Stable `IterableDataset`, `get_worker_info()` |
| pyarrow | >=15.0 | All file format reading, predicate pushdown |
| fsspec | >=2024.2 | Filesystem abstraction |

### Optional

| Extra | Libraries | Install |
|-------|-----------|---------|
| `s3` | `s3fs>=2024.2` | `pip install torch-dataloader-utils[s3]` |
| `gcs` | `gcsfs>=2024.2` | `pip install torch-dataloader-utils[gcs]` |
| `azure` | `adlfs>=2024.2` | `pip install torch-dataloader-utils[azure]` |
| `iceberg` | `pyiceberg>=0.6` | `pip install torch-dataloader-utils[iceberg]` |
| `all` | all of the above | `pip install torch-dataloader-utils[all]` |

Missing optional dependencies raise a clear `ImportError` with the install command at the point of use.

---

## Challenges

### Worker Count at Split Time
Splits must be generated before the DataLoader starts, but `num_workers` is a DataLoader concern. Solved by making `create_dataloader()` the single entry point — it owns both split generation and DataLoader construction. `num_workers=None` auto-detects using `os.cpu_count() - 1`.

### Split Generation Timing
File discovery is expensive (S3 scan) and happens once at `create_dataloader()`. Split generation is cheap (list operations) and happens at construction — only regenerated when `set_epoch()` is called with shuffle enabled.

### Iceberg Splits Need Richer Metadata
Plain file splits only need paths. Iceberg enables smarter splitting using partition values and column statistics from the snapshot manifest. Solved with `DataFileInfo` (base, for plain files) and `IcebergDataFileInfo` (extends base with Iceberg metadata). The `SplitStrategy` protocol takes `list[DataFileInfo]` — works for both via inheritance.

### Arrow Batching vs DataLoader Batching
`pyarrow.dataset` reads in `RecordBatch` chunks internally. If the DataLoader also batches, Arrow output gets re-batched — redundant and wasteful. Solved by always constructing `DataLoader(batch_size=None)` and letting `batch_size` on the dataset control Arrow read size.

### Epoch Reshuffling in IterableDataset
`IterableDataset` has no explicit epoch boundary hook. Reshuffling is triggered by `set_epoch(n)` in the main process — regenerates splits with `seed + epoch` and workers receive the updated split list via pickling at the next epoch start.

### Uneven Worker Load
Round-robin file assignment produces splits with equal file counts but uneven work if files vary in size. Solved by `TargetSizeSplitStrategy`: packs Parquet row groups into target-sized chunks and distributes them using greedy min-heap (LPT scheduling), which always assigns the next chunk to the least-loaded worker. For unequal files, this produces near-perfectly balanced row counts across workers even when split counts differ.

### Iceberg Partition Pruning
Pyarrow `pc.Expression` predicates (`filters`) are applied as row-level filters after reading — all files in a partition are scanned. For coarse partition pruning before any data is read, use `scan_filter` with a native pyiceberg expression (`GreaterThan`, `LessThan`, etc.). This is pushed into `table.scan(row_filter=...)` at `plan_files()` time, excluding non-matching partitions and files before splits are generated. Both `filters` and `scan_filter` can be used together.

### Iceberg Delete Files
Iceberg supports row-level deletes via **position delete files** (written by `DELETE` and `MERGE INTO` operations). Reading Parquet data files directly would return deleted rows — pyarrow has no knowledge of Iceberg delete files.

**Solution:** `IcebergDataset` inspects every `FileScanTask` from `scan.plan_files()` at construction time. When delete files are present, it switches to `pyiceberg.io.pyarrow.ArrowScan` per file task — ArrowScan applies position deletes before yielding batches. When no delete files exist, it uses the direct pyarrow reader (faster, with sub-file row-range splitting active).

**Limitations:**
- **Equality deletes** are not supported by pyiceberg and will raise `NotImplementedError`. Compact the table first.
- **Sub-file splitting is disabled** when delete files are present — position delete offsets reference absolute row positions in the original file, not in a sub-range slice.

### Iceberg Catalog Diversity
Iceberg supports REST, Glue, Hive, and JDBC catalogs — each with different config shapes. The library passes `catalog_config` directly to `pyiceberg` without interpretation, so catalog compatibility is delegated to `pyiceberg` rather than re-implemented here.

### No Record-Level Shuffle
True record-level shuffle requires either loading all data into memory or maintaining a large shuffle buffer. V1 does not support this. File-level and chunk-level shuffle is sufficient for most training workloads.

### Credential Lifetime and Token Refresh
`storage_options` is a plain dict — it is pickled and sent to each worker at DataLoader startup. Credentials that expire mid-training (AWS STS session tokens, OAuth2 tokens, Vault-issued dynamic credentials) will cause worker failures when they expire. This library provides no credential refresh mechanism. Users are responsible for ensuring credentials remain valid for the duration of training — for example by using IAM roles, managed identities, or a process-level credential refresh daemon (`k5start` for Kerberos, `aws-vault` for STS). Ambient credentials (IAM instance profiles, GKE workload identity, Azure managed identity) are immune to this issue and are the recommended approach for long-running training jobs.

---

## V1 Scope

- File-based formats: Parquet, ORC, CSV, JSON, JSONL
- Iceberg table support via `pyiceberg`
- S3, GCS, Azure, local filesystem support via `fsspec`
- `RoundRobinSplitStrategy` — file-count balanced splits (fallback)
- `TargetSizeSplitStrategy` — target-sized sub-file splits with LPT scheduling (default)
- Sub-file row-range splitting for Parquet via `RowRange` — large files distributed across workers at row group granularity
- `split_bytes` / `split_rows` parameters for tuning chunk size
- Auto-selection of split strategy based on file list
- Pluggable `SplitStrategy` protocol — user-defined strategies, no inheritance needed
- `DataFileInfo` and `IcebergDataFileInfo` for rich file metadata
- File-level shuffle with epoch reshuffling via `set_epoch()`
- Column projection and predicate pushdown via `pyarrow` and `pyiceberg`
- `scan_filter` for Iceberg scan-level partition/file pruning via pyiceberg expressions
- Hive partitioning decoding via `partitioning="hive"`
- Output formats: `torch`, `numpy`, `arrow`, `dict`
- `collate_fn` passthrough with early validation for non-torch output formats
- `num_workers=None` auto-detection
- `create_dataloader()` as the single entry point returning `(DataLoader, dataset)`
- PyPI publishing
- Docs setup

## V2 Scope

- Record-level shuffle via configurable shuffle buffer
- Row-level interleaving across files within a split
- Mid-epoch checkpoint and resume — persist which splits have been fully consumed so that on crash/restart the DataLoader can skip already-processed splits and resume from the partial one; epoch number checkpointed alongside model weights for deterministic shuffle resumption
- Accelerate-native integration — rank-aware split assignment so each DDP rank gets a disjoint set of splits with no cross-rank data overlap; `accelerator` parameter on `create_dataloader()`
- Horovod and DeepSpeed rank-aware split assignment (same mechanism as Accelerate)
- GCS and Azure real backend CI tests — S3 (moto) covers the shared fsspec/PyFileSystem code path in V1; V2 adds Docker Compose-based GCS (fake-gcs-server) and Azure (Azurite) CI tests to catch per-backend auth, path format, and stat() response differences
- Real multi-worker DataLoader integration tests on Linux CI — macOS spawn mode causes deadlocks with pyarrow generators; Linux fork mode works correctly
- Metrics: rows read, bytes read, worker utilization
- Build stateful dataloader state_dict() / load_state_dict() to checkpointing/resuming a crash
- ORC support for Iceberg

## V3 Scope

- Adaptive dynamic splitting based on runtime throughput — rebalance splits across workers during iteration if some workers finish significantly faster than others

---

## Publishing

Package name: `torch-dataloader-utils`
Import name: `torch_dataloader_utils`
Registry: PyPI
Versioning: semver (`MAJOR.MINOR.PATCH`)

CI tests the four corners of the support matrix:

| | Python 3.11 | Python 3.12 |
|-|-------------|-------------|
| PyTorch 2.2 | ✓ | ✓ |
| PyTorch latest | ✓ | ✓ |
