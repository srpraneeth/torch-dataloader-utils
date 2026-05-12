# Distributed Data Loading

This page covers rank-aware file sharding for multi-GPU and multi-node training. If you are running a single process, you do not need any of this — the defaults (`num_ranks=1, rank=0`) give you identical behaviour to V1.

---

## Two-Level Hierarchy

The library partitions data at two independent levels:

```
                   All files / chunks
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       Rank 0          Rank 1          Rank 2       ← rank partitioning
          │               │               │            (num_ranks / rank)
     ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
  Worker 0  Worker 1  Worker 0  Worker 1  Worker 0  Worker 1
                                                    ← worker partitioning
                                                       (num_workers)
```

**Rank partitioning** — which file chunks each DDP process is responsible for. Computed once in the main process before any worker spawns.

**Worker partitioning** — how each rank's chunks are further divided across its local I/O workers. Unchanged from single-process behaviour.

The result: each file chunk is owned by exactly one rank and exactly one worker. No file is ever read twice, regardless of how many ranks or workers you use.

---

## How Rank Partitioning Works

After the split strategy generates all chunks, they are assigned to ranks using an interleaved slice:

```python
rank_splits = all_splits[rank::num_ranks]
```

With `num_ranks=3`:

```
all_splits:   [C0, C1, C2, C3, C4, C5, C6, C7, C8]

Rank 0 gets:  [C0, C3, C6]   (indices 0, 3, 6)
Rank 1 gets:  [C1, C4, C7]   (indices 1, 4, 7)
Rank 2 gets:  [C2, C5, C8]   (indices 2, 5, 8)
```

Interleaved assignment distributes large and small chunks evenly across ranks without any sorting by rank. With shuffle enabled, shuffling happens before slicing — so each rank sees a different random subset each epoch, but together they always cover all data.

Each rank then assigns its own slice to its `num_workers` I/O workers using the same greedy LPT heap used in single-rank mode.

---

## PyTorch DDP

```python
import torch.distributed as dist
from torch_dataloader_utils import StructuredDataset

dist.init_process_group(backend="nccl")

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/training-data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    shuffle_seed=42,
    num_ranks=dist.get_world_size(),   # e.g. 8 for 8-GPU job
    rank=dist.get_rank(),              # 0–7
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)           # must be called in main process
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch["feature_a"], batch["label"])
        loss.backward()
        optimizer.step()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 train.py
```

---

## HuggingFace Accelerate

```python
from accelerate import Accelerator
from torch_dataloader_utils import StructuredDataset

accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/training-data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=accelerator.num_processes,
    rank=accelerator.process_index,    # global rank — NOT local_process_index
)

# accelerator.prepare() is optional here — adds gradient sync wrappers
# but does NOT re-shard data (this library already handles that)
loader = accelerator.prepare(loader)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch["feature_a"], batch["label"])
        accelerator.backward(loss)
        optimizer.step()
```

---

## Horovod

```python
import horovod.torch as hvd
from torch_dataloader_utils import StructuredDataset

hvd.init()
torch.cuda.set_device(hvd.local_rank())

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/training-data/",
    format="parquet",
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    num_ranks=hvd.size(),
    rank=hvd.rank(),      # global rank
)

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch["feature_a"], batch["label"])
        loss.backward()
        optimizer.step()
```

---

## Iceberg Tables with DDP

`IcebergDataset` accepts the same `num_ranks` and `rank` parameters:

```python
import pyarrow.compute as pc
import torch.distributed as dist
from torch_dataloader_utils import IcebergDataset

dist.init_process_group(backend="nccl")

loader, dataset = IcebergDataset.create_dataloader(
    table="my_db.events",
    catalog_config={
        "type": "glue",
        "region_name": "us-east-1",
    },
    num_workers=4,
    batch_size=1024,
    shuffle=True,
    filters=pc.field("region_id") >= 5,
    num_ranks=dist.get_world_size(),
    rank=dist.get_rank(),
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

!!! note "Delete files and rank sharding"
    When the Iceberg table has delete files, sub-file splitting is disabled and each file is assigned as a whole chunk. Rank partitioning still works — each rank reads a disjoint subset of files. See [Limitations](limitations.md) for details.

---

## set_epoch() in Distributed Training

With `shuffle=True`, call `set_epoch(epoch)` **in the main process of every rank** before each epoch. All ranks must use the same seed and epoch number — this is guaranteed as long as every process calls `set_epoch(epoch)` with the same `epoch` value.

```python
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)   # call on every rank
    for batch in loader:
        ...
```

The shuffle is applied to the full chunk list before rank slicing, so the assignment is globally consistent: if rank 0 shuffled differently from rank 1, they could overlap.

With `shuffle=False`, `set_epoch()` can be omitted — splits are always generated in the same deterministic order regardless of epoch number.

---

## Global Rank vs Local Rank

!!! warning "Always use global rank"
    `rank` must be the **global** rank — unique across all processes in the entire job (0 to `world_size - 1`). Never pass the local rank (per-node GPU index).

    On a 2-node × 4-GPU job (`world_size=8`):

    | Node | GPU | Global rank | Local rank |
    |------|-----|-------------|------------|
    | 0 | 0 | 0 | 0 |
    | 0 | 1 | 1 | 1 |
    | 0 | 2 | 2 | 2 |
    | 0 | 3 | 3 | 3 |
    | 1 | 0 | 4 | 0 |
    | 1 | 1 | 5 | 1 |
    | 1 | 2 | 6 | 2 |
    | 1 | 3 | 7 | 3 |

    Using local rank would cause ranks 0 and 4 to both request rank 0's data — duplicating reads and missing data on ranks 4–7.

| Framework | Global rank | World size |
|-----------|-------------|------------|
| PyTorch DDP | `dist.get_rank()` | `dist.get_world_size()` |
| Accelerate | `accelerator.process_index` | `accelerator.num_processes` |
| Horovod | `hvd.rank()` | `hvd.size()` |
| Environment | `int(os.environ["RANK"])` | `int(os.environ["WORLD_SIZE"])` |

---

## Edge Cases

### More ranks than chunks

If `num_ranks > len(all_splits)`, some ranks receive zero splits and yield nothing. This is valid — those ranks simply have an empty epoch. Example with 2 chunks and 4 ranks:

```
Rank 0 → [C0]     # 1 chunk
Rank 1 → [C1]     # 1 chunk
Rank 2 → []       # empty — yields nothing this epoch
Rank 3 → []       # empty — yields nothing this epoch
```

### Uneven split count across ranks

With 7 chunks and 3 ranks, ranks get 3, 2, 2 chunks respectively — differing by at most 1. This is an inherent property of interleaved slicing and cannot be avoided without padding.

### Custom V1 strategies

Strategies that implement only the V1 `generate(files, num_workers, epoch)` signature still work — the library detects the missing `num_ranks`/`rank` params via `inspect.signature` and falls back to the three-argument call. Those strategies receive all chunks regardless of rank.

---

## Checklist

Before running a distributed job:

- [ ] Pass `num_ranks=world_size` and `rank=global_rank` to `create_dataloader()`
- [ ] Use global rank (not local rank)
- [ ] Call `dataset.set_epoch(epoch)` on **every rank** at the start of each epoch when `shuffle=True`
- [ ] Verify `len(all_files) >= num_ranks` — if not, some ranks will be empty
- [ ] For Iceberg: ensure the catalog is reachable from every rank's host
