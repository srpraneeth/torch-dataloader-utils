# torch-dataloader-utils

A lightweight PyTorch library for reading structured tabular data from cloud object storage directly into a `DataLoader`. No Ray, no Spark, no heavy infrastructure.

---

## Why?

PyTorch has no good built-in story for streaming structured data from cloud storage into a `DataLoader`. The existing ecosystem is either deprecated, too heavy, or built for different data shapes:

| Solution | Problem |
|----------|---------|
| `torchdata` | Solves checkpointing (`StatefulDataLoader`) and pipeline composition (`nodes`); no built-in cloud file sharding or Iceberg support — requires significant custom wiring |
| Ray Data | Requires a Ray cluster — overkill for most training jobs |
| HuggingFace `datasets` | No Iceberg support, opinionated schema |
| WebDataset | Designed for unstructured data (images, audio) |

This library fills that gap. But to understand why it exists, it helps to understand exactly where `torch.utils.data` falls short.

### Problem 1 — Map-style Dataset requires indexable data

PyTorch's canonical pattern uses a map-style `Dataset`:

```python
class MyDataset(Dataset):
    def __len__(self): ...         # must know total row count
    def __getitem__(self, idx): ...  # must seek to any row by index
```

This works well for in-memory data or image folders. It **breaks for cloud-stored tabular data** because:

- **`__len__` is expensive** — you'd need to scan metadata across potentially thousands of Parquet files just to return a number
- **`__getitem__(idx)` has poor granularity** — Parquet supports seeking to a *row group* (typically 100k–1M rows) via footer metadata, but not to an individual row. Fetching row 1,234,567 means decoding the entire row group it lives in — potentially hundreds of MBs — to return one row
- **`DistributedSampler` destroys I/O locality** — it generates a globally shuffled list of row indices and assigns each rank a scattered slice. Say you have 3 files with 3 row groups each (9 row groups total) and 2 DDP ranks:

    ```
    DistributedSampler assigns:
      Rank 0 → rows [0, 2, 4, 6, 8, 10, 12, 14, 16] (every other row — from ALL row groups)
      Rank 1 → rows [1, 3, 5, 7, 9, 11, 13, 15, 17] (every other row — from ALL row groups)

    To serve those rows, both ranks must open all 3 files
    and decode all 9 row groups — just to use half the rows from each.
    Every row group is decoded twice. Total I/O: 2×.
    ```

    On cloud storage, each row group seek is a separate HTTP range request (~50–100 ms latency each). Sequential reads are fast; scattered seeks across row groups are slow and expensive. The more ranks you add, the worse the amplification.

### Problem 2 — IterableDataset has no distributed sharding

`IterableDataset` drops the index contract, which fits file streaming naturally. But PyTorch gives workers **no mechanism to know which files belong to them**:

```python
class MyIterableDataset(IterableDataset):
    def __iter__(self):
        # get_worker_info() tells you your worker id and num_workers
        # but PyTorch never tells you *which files* are yours
        for file in all_files:
            yield from read(file)
```

The standard workaround: every worker reads every file and filters by index parity.

```python
worker = torch.utils.data.get_worker_info()
for i, row in enumerate(all_rows):
    if i % worker.num_workers == worker.id:
        yield row   # keep 1-in-N rows, discard the rest
```

Every file is read N times. On S3 or GCS, you pay per-byte — including bytes you immediately throw away.

### Problem 3 — Accelerate / DDP gives no file-level sharding either

HuggingFace Accelerate and PyTorch DDP wrap your DataLoader across GPU ranks. But wrapping an `IterableDataset`-based loader doesn't partition the underlying files — it just splits the **batch stream** after data has already been loaded. Each rank still runs its own DataLoader that reads the full dataset:

```
Rank 0: DataLoader → reads ALL files → sends every other batch to GPU 0
Rank 1: DataLoader → reads ALL files → sends every other batch to GPU 1
→ full dataset read twice, twice the I/O cost
```

Accelerate's documentation acknowledges this and says "implement sharding yourself" — which is exactly what this library does.

### What this library does

Instead of distributing *row indices* after data is loaded, this library distributes *files* (or row group ranges for large Parquet files) **before iteration begins**. Same 3-file, 2-rank example:

```
This library assigns at create_dataloader() time:
  Rank 0 → File A (all row groups) + File B (all row groups)
  Rank 1 → File C (all row groups)

Each file is opened once, read sequentially start to finish,
by exactly one worker. No row group is ever decoded twice.
Total I/O: 1×.
```

**What is a split?**

A *split* (called a `Shard` internally) is a worker's read assignment — the list of file chunks it is responsible for. A chunk is either a whole file or a row group range within a file (`RowRange(offset, length)`). How chunks are generated depends on the format:

| Format | Chunk granularity | Sub-file splitting |
|--------|-------------------|-------------------|
| Parquet | Row group (from footer metadata) | Yes — one file → many chunks |
| Iceberg | Resolves to Parquet data files → same as Parquet | Yes |
| ORC | Whole file | No — one file = one chunk |
| CSV / JSON / JSONL | Whole file | No — one file = one chunk |

For Parquet and Iceberg, footer metadata is scanned once in the main process (no data read) to determine row group sizes, then row groups are packed into target-sized chunks. For ORC and text formats, each file becomes one unsplittable chunk — so for good parallelism, shard large ORC/CSV files into many smaller ones.

```
Parquet example — 3 files with different sizes:

File A: 3 row groups  →  chunk A1 (rg 0-1), chunk A2 (rg 2)
File B: 2 row groups  →  chunk B1 (rg 0-1)
File C: 4 row groups  →  chunk C1 (rg 0-1), chunk C2 (rg 2-3)

All chunks: [A1, A2, B1, C1, C2]
  → shuffle (optional)
  → round-robin assign to workers:

Worker 0 Shard: [A1, C1]   ← reads row groups 0-1 of A, then 0-1 of C
Worker 1 Shard: [A2, C2]   ← reads row group 2 of A, then 2-3 of C
Worker 2 Shard: [B1]       ← reads row groups 0-1 of B
```

The assignment is serialized into each worker's initializer — workers receive their shard before they start and never need to communicate with each other or the main process during iteration.

See [Splits & Workers](splits.md) for full details on split strategies, sub-file splitting, shuffle, and tuning chunk size.

- **File-level sharding at startup** — splits are assigned once in the main process at `create_dataloader()` time, before any worker spawns. Workers never coordinate.
- **Sub-file splitting for large Parquet** — a single 10 GB file can be split across multiple workers at row group boundaries; each worker reads a disjoint range sequentially.
- **Each byte read exactly once** — no worker reads a file another worker owns. No amplification regardless of how many workers or ranks you use.
- **Standard `DataLoader` output** — returns a plain `torch.utils.data.DataLoader`. Nothing else in your training stack changes. Works with Accelerate, FSDP, DDP unchanged.
- **Epoch-level shuffle** — chunks are shuffled before assignment each epoch via `dataset.set_epoch(epoch)`, giving good randomness without record-level I/O overhead.

## One Interface Across Everything

A key design goal is that **where your data lives and how it is stored should not change your training code**. The library abstracts three independent axes:

```
Storage backend     Format              Catalog
──────────────      ──────────────      ──────────────
S3                  Parquet             —
GCS          ×      ORC          ×      Apache Iceberg
Azure               CSV                 (REST, Glue,
Local               JSON / JSONL         Hive, JDBC)
                                        ──────────────
        Any combination → same DataLoader interface
```

Whether you are reading raw Parquet files from S3, ORC files from GCS, or an Iceberg table in AWS Glue — the call looks the same and the output is the same:

```python
# Raw files on S3
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/", format="parquet", ...
)

# Raw files on GCS
loader, dataset = StructuredDataset.create_dataloader(
    path="gcs://bucket/data/", format="orc", ...
)

# Iceberg table via Glue catalog
loader, dataset = IcebergDataset.create_dataloader(
    table="my_db.my_table",
    catalog_config={"type": "glue", "region_name": "us-east-1"},
    ...
)

# All three return the same standard DataLoader
for batch in loader:
    loss = model(batch["feature_a"], batch["label"])
```

**How this is achieved:**

- **`fsspec`** provides a unified filesystem interface — S3, GCS, Azure, and local all expose the same `open()` / `ls()` API. Adding a new cloud provider requires only installing the right `fsspec` plugin (`s3fs`, `gcsfs`, `adlfs`).
- **`pyarrow`** provides a unified reader across Parquet, ORC, CSV, and JSON with consistent predicate pushdown, column projection, and Arrow `RecordBatch` output regardless of the underlying format.
- **`pyiceberg`** resolves Iceberg tables to their underlying data files — which are just Parquet files — and hands them to the same reader pipeline. Partition pruning, snapshot pinning, and delete file handling are all transparent.

The result: switching from local Parquet during development to Iceberg-on-S3 in production is a one-line config change, not a rewrite.

---

## Install

```bash
pip install torch-dataloader-utils          # core (Parquet, ORC, CSV, JSON)
pip install torch-dataloader-utils[s3]      # + S3 support
pip install torch-dataloader-utils[iceberg] # + Apache Iceberg support
pip install torch-dataloader-utils[all]     # everything
```

---

## Quick Start

```python
import pyarrow.compute as pc
from torch_dataloader_utils import StructuredDataset

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://my-bucket/data/train/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    columns=["feature_a", "feature_b", "label"],
    filters=pc.field("date") > "2024-01-01",
    shuffle=True,
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        loss = model(batch["feature_a"], batch["label"])
        loss.backward()
```

For Iceberg tables:

```python
from torch_dataloader_utils import IcebergDataset

loader, dataset = IcebergDataset.create_dataloader(
    table="my_db.my_table",
    catalog_config={"type": "rest", "uri": "https://catalog.example.com"},
    num_workers=4,
    batch_size=1024,
    filters=pc.field("region_id") >= 5,   # auto-prunes files + filters rows
)
```

---

## Architecture

```
FilesystemLayer   →   FormatLayer          →   DatasetLayer
fsspec                pyarrow.dataset           torch.IterableDataset
                      pyiceberg (Iceberg only)
```

Files are discovered once at `create_dataloader()` time, partitioned into per-worker splits, and read lazily during iteration. `DataLoader` is always constructed with `batch_size=None` — Arrow owns batching internally.

---

## Dependencies

### Core

| Library | Version | Role |
|---------|---------|------|
| PyTorch | ≥2.2 | `IterableDataset`, `get_worker_info()` |
| pyarrow | ≥15.0 | All file format reading, predicate pushdown |
| fsspec  | ≥2024.2 | Filesystem abstraction |

### Optional

| Extra | Install |
|-------|---------|
| `s3` | `pip install torch-dataloader-utils[s3]` |
| `gcs` | `pip install torch-dataloader-utils[gcs]` |
| `azure` | `pip install torch-dataloader-utils[azure]` |
| `iceberg` | `pip install torch-dataloader-utils[iceberg]` |
| `all` | `pip install torch-dataloader-utils[all]` |
