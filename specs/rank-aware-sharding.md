# Spec: Rank-Aware DDP Sharding

## Core Principle

In distributed training (DDP, FSDP, Horovod, DeepSpeed) multiple **ranks** run as separate
processes, each with its own DataLoader. Without rank awareness every rank discovers the same
files and generates the same splits — every rank reads all the data, producing duplicate
gradient updates and wasting network bandwidth.

Rank-aware sharding adds a second level of partitioning **above** the per-worker split layer:

```
Files
  └─ rank 0 gets files [0, 2, 4, ...]        ← rank-level partition (new)
  └─ rank 1 gets files [1, 3, 5, ...]

  Within each rank:
    └─ worker 0 gets splits [0, 2, ...]       ← worker-level partition (existing)
    └─ worker 1 gets splits [1, 3, ...]
```

The two levels are independent:
- **Rank partitioning** — which files (or sub-file chunks) each rank is responsible for.
  Computed once in the main process before workers fork.
- **Worker partitioning** — how each rank's files are further divided across its local workers.
  Unchanged from V1.

---

## Design

### Split Pipeline with Rank Sharding

```
discover_files()
    → strategy.generate(files, num_workers=W, num_ranks=R, rank=r, epoch=E)
        → generate all splits as flat list (same as V1)
        → partition splits by rank  (new — slice [rank::num_ranks] or contiguous block)
        → assign rank's slice to W workers via LPT heap (same as V1)
        → return W Shards
```

`generate()` is the single point of change. `__iter__`, `read_split`, and the reader layer
are untouched.

### `SplitStrategy.generate()` — Updated Protocol

```python
class SplitStrategy(Protocol):
    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int = 0,
        num_ranks: int = 1,   # total DDP ranks — new
        rank: int = 0,        # this process's rank — new
    ) -> list[Shard]:
        ...
```

Default values (`num_ranks=1`, `rank=0`) preserve V1 behaviour — no breaking change.

### Rank Partitioning Algorithm

After generating the flat split list (post-shuffle or post-LPT-sort), slice it by rank:

```python
rank_splits = all_splits[rank::num_ranks]   # interleaved — round-robin across ranks
```

Interleaved assignment is preferred over contiguous blocks because:
- It distributes large and small splits evenly across ranks without sorting by rank.
- With shuffle enabled, interleaved assignment over a shuffled list is equivalent to
  independently shuffled per-rank lists.
- It is a one-liner with no state.

After slicing, assign `rank_splits` to `num_workers` workers via the existing LPT heap.

### `StructuredDataset` / `IcebergDataset` — New Parameters

```python
StructuredDataset(
    ...
    num_ranks: int = 1,    # total DDP world size
    rank: int = 0,         # this process's rank (0-indexed)
)
```

`create_dataloader()` gains the same two parameters and passes them through to
`strategy.generate()`.

### Accelerate / Horovod / DeepSpeed Integration

Users can pass rank info from any distributed backend:

```python
# PyTorch DDP
import torch.distributed as dist
rank, num_ranks = dist.get_rank(), dist.get_world_size()

# Accelerate
from accelerate import Accelerator
acc = Accelerator()
rank, num_ranks = acc.process_index, acc.num_processes

# Horovod
import horovod.torch as hvd
rank, num_ranks = hvd.rank(), hvd.size()
```

> **Global rank vs local rank** — `rank` here is always the **global rank** (unique across
> the entire job, 0 to `world_size - 1`). Do not pass `local_rank` — that is the per-node
> GPU index used only for `torch.cuda.set_device()`. On a 2-node × 4-GPU job, global ranks
> are 0–7 and local ranks are 0–3 on each node. Data partitioning must use global rank so
> each GPU process receives a disjoint file slice regardless of which node it runs on.

### Shuffle Across Ranks

With `shuffle=True`, the flat split list is shuffled with `seed + epoch` before slicing by
rank. This means:

- All ranks see the same shuffled order.
- Interleaved slicing distributes that order evenly.
- With the same `seed` and `epoch`, all ranks produce deterministic, disjoint assignments.

No per-rank seed offset is needed — the interleaved slice ensures disjointness regardless
of shuffle state.

### Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| `num_ranks=1` (default) | Identical to V1 — all splits go to rank 0 |
| `rank >= num_splits` | Rank gets 0 splits — `__iter__` yields nothing |
| `num_ranks > num_splits` | Some ranks get 0 splits |
| `num_workers > rank_splits` | Some workers within the rank get 0 splits (existing V1 behaviour) |
| Unequal split count across ranks | Differs by at most 1 split (interleaved assignment property) |

---

## Requirements

### New Parameters `[v2]`
The system SHALL accept `num_ranks: int = 1` and `rank: int = 0` on `StructuredDataset`,
`IcebergDataset`, and `create_dataloader()`.
The system SHALL pass both parameters through to `strategy.generate()`.
The system SHALL raise `ValueError` if `rank >= num_ranks`.
The system SHALL raise `ValueError` if `num_ranks < 1`.

### Rank Partitioning `[v2]`
The system SHALL partition splits across ranks using interleaved (round-robin) assignment:
`rank_splits = all_splits[rank::num_ranks]`.
Partitioning SHALL happen after shuffle (if enabled) and before worker assignment.
No split SHALL appear in more than one rank's assignment.
The union of all ranks' splits SHALL equal the full split list.

### Worker Assignment `[v2]`
After rank partitioning, worker assignment SHALL use the existing LPT heap over `rank_splits`.
`num_workers` refers to workers **per rank** — not total workers across all ranks.

### Backward Compatibility `[v2]`
Default `num_ranks=1`, `rank=0` SHALL produce identical output to V1 `generate()`.
All existing `SplitStrategy` implementations SHALL continue to work without modification
until they explicitly add the new parameters.

### Determinism `[v2]`
Given the same `seed`, `epoch`, `num_ranks`, and `rank`, split assignment SHALL be identical
across independent process restarts.
Two ranks with the same `seed` and `epoch` SHALL receive disjoint split sets.

---

## API Example

```python
import torch.distributed as dist
from torch_dataloader_utils.dataset.structured import StructuredDataset

dist.init_process_group("nccl")

loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/training-data/",
    format="parquet",
    batch_size=1024,
    shuffle=True,
    shuffle_seed=42,
    num_workers=4,
    num_ranks=dist.get_world_size(),   # e.g. 8 for 8-GPU job
    rank=dist.get_rank(),              # 0–7
)

for epoch in range(10):
    dataset.set_epoch(epoch)
    for batch in loader:
        ...
```

---

## Files to Change

| File | Change |
|------|--------|
| `src/torch_dataloader_utils/splits/core.py` | Add `num_ranks`, `rank` to `SplitStrategy` protocol |
| `src/torch_dataloader_utils/splits/target_size.py` | Add rank partitioning slice to `generate()` |
| `src/torch_dataloader_utils/splits/round_robin.py` | Add rank partitioning slice to `generate()` |
| `src/torch_dataloader_utils/dataset/structured.py` | Accept + pass through `num_ranks`, `rank` |
| `src/torch_dataloader_utils/dataset/iceberg.py` | Accept + pass through `num_ranks`, `rank` |
| `tests/unit/splits/test_rank_sharding.py` | New — unit tests for rank partitioning |
| `tests/integration/test_local.py` | Add multi-rank scenario (simulated via direct generate() calls) |
| `specs/rank-aware-sharding.md` | This file |

---

## Scenarios

**Rank distribution** — interleaved slicing, `num_workers=2` per rank

| `num_ranks` | `rank` | Total splits | Rank gets | Workers get |
|-------------|--------|-------------|-----------|-------------|
| 1 | 0 | 4 | all 4 (V1 identical) | 2 each |
| 2 | 0 | 8 | splits [0,2,4,6] | 2 each |
| 2 | 1 | 8 | splits [1,3,5,7] | 2 each |
| 3 | 0 | 7 | 3 splits | — |
| 3 | 1 | 7 | 2 splits | — |
| 3 | 2 | 7 | 2 splits | — |
| 4 | 0 | 2 | 1 split | — |
| 4 | 2 | 2 | 0 splits — yields nothing | — |

**Correctness invariants**
- Union of all ranks' splits = full split list, no overlap
- `num_ranks=1, rank=0` output is byte-for-byte identical to V1 `generate()` with no rank params

**Shuffle determinism** — `shuffle=True, seed=42, epoch=0, num_ranks=2`: rank 0 and rank 1 called independently → disjoint split sets; same call repeated → identical assignment

**Epoch variation** — `shuffle=True, num_ranks=2`: epoch 0 vs epoch 1 → different split assignments on both ranks

**Validation errors**

| Condition | Error |
|-----------|-------|
| `rank=2, num_ranks=2` | `ValueError` |
| `num_ranks=0` | `ValueError` |
