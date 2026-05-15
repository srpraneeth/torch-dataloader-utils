# Checkpoint and Resume

Save `dataset.state_dict()` alongside your model checkpoint. On resume, call
`dataset.load_state_dict()` before the DataLoader begins iterating — workers whose shards
were already completed return immediately without touching the filesystem.

---

## Saving a checkpoint

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,
    shuffle_seed=42,
    num_workers=8,
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for step, batch in enumerate(loader):
        loss = model(batch)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            torch.save({
                "model":   model.state_dict(),
                "dataset": dataset.state_dict(),
                "epoch":   epoch,
                "step":    step,
            }, f"ckpt_{epoch}_{step}.pt")
```

---

## Resuming

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,
    shuffle_seed=42,    # must match the original run
    num_workers=8,      # must match the original run
)

start_epoch = 0
if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    dataset.load_state_dict(ckpt["dataset"])  # restores epoch + skips completed shards
    start_epoch = ckpt["epoch"]

for epoch in range(start_epoch, num_epochs):
    if epoch != start_epoch:
        dataset.set_epoch(epoch)   # do NOT call for the resumed epoch
    for batch in loader:
        ...
```

!!! warning "Do not call `set_epoch()` for the resumed epoch"
    `load_state_dict()` sets the epoch and regenerates splits internally. Calling
    `set_epoch()` afterwards clears `_completed_workers` and defeats the resume.

---

## What state_dict() contains

```python
{
    "epoch": 3,
    "_num_workers": 8,       # stored for mismatch diagnosis
    "_shuffle_seed": 42,     # stored for mismatch diagnosis
    "completed_shards": [
        {
            "splits": [
                {"path": "s3://bucket/part-0001.parquet", "row_offset": 0,    "row_length": 250000},
                {"path": "s3://bucket/part-0002.parquet", "row_offset": None, "row_length": None},
            ]
        },
        ...
    ]
}
```

The state stores **shard content** — file paths and row ranges — not worker IDs. This means
`load_state_dict()` validates by comparing actual split content rather than trusting that
worker ID assignments are unchanged. `row_offset=None, row_length=None` means the whole file
(no sub-file split).

---

## CheckpointMismatchError

`load_state_dict()` raises `CheckpointMismatchError` if any stored shard cannot be matched
against the current splits. This happens when `num_workers`, `shuffle_seed`, or the file
list changed between the checkpoint and the resume:

```
CheckpointMismatchError: Checkpoint shard does not match any current split.

  Checkpoint shard:
    part-0001.parquet  rows [0, 250,000)
    part-0002.parquet  full file

  Likely cause: num_workers changed: checkpoint=8, current=4

  Reconstruct the dataset with matching parameters or discard this checkpoint.
```

Catch it explicitly if you want to fall back to a fresh epoch:

```python
from torch_dataloader_utils import CheckpointMismatchError

try:
    dataset.load_state_dict(ckpt["dataset"])
except CheckpointMismatchError as e:
    logging.warning("Checkpoint incompatible, starting epoch from scratch: %s", e)
    start_epoch = 0
```

---

## Re-processing on resume

Completed shards are skipped with **zero I/O**. The one shard that was in-progress at crash
time re-reads from its start — at most `split_bytes` worth of data (128 MiB by default).
With 8 workers this is at most 12.5% of one epoch.

Compare this to the common alternative of checkpointing `(epoch, step)` and fast-forwarding
on resume by reading and discarding data — that approach re-reads everything up to the
checkpoint step regardless of how far along the epoch was.

---

## DDP / Multi-rank

Model weights are identical across ranks so saving from rank 0 is sufficient. Dataset state
is **not** — each rank processed a different subset of shards (rank 0 gets splits 0, 4, 8...;
rank 1 gets splits 1, 5, 9...). You cannot use rank 0's dataset state to resume rank 1 — it
would mark the wrong shards as completed.

`state_dict()` and `load_state_dict()` are rank-local operations — the library does not call
any `dist.*` functions internally. Coordinating across ranks is the training loop's
responsibility, consistent with how `model.state_dict()` works.

### Option A — Single checkpoint file (recommended)

Gather all ranks' dataset states onto rank 0, save once. On resume each rank extracts its
own slice. One file, no per-rank bookkeeping.

```python
import torch.distributed as dist

# Save — gather on rank 0, save once
dataset_state = dataset.state_dict()
all_dataset_states = [None] * dist.get_world_size()
dist.all_gather_object(all_dataset_states, dataset_state)

if dist.get_rank() == 0:
    torch.save({
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "dataset_states": all_dataset_states,   # list indexed by rank
        "epoch":          epoch,
        "step":           step,
    }, "checkpoint.pt")

# Resume — load on rank 0, broadcast, each rank takes its slice
if dist.get_rank() == 0:
    ckpt = torch.load("checkpoint.pt", weights_only=False)
else:
    ckpt = None
ckpt_list = [ckpt]
dist.broadcast_object_list(ckpt_list, src=0)
ckpt = ckpt_list[0]

model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
dataset.load_state_dict(ckpt["dataset_states"][dist.get_rank()])
```

### Option B — Per-rank files (simpler, more files)

Each rank saves and loads its own checkpoint independently. No distributed collectives
required, but you get `num_ranks` checkpoint files to manage.

```python
# Save — every rank writes its own file
torch.save(
    {"model": model.state_dict(), "dataset": dataset.state_dict(), ...},
    f"ckpt_rank{dist.get_rank()}_{step}.pt",
)

# Resume — every rank reads its own file
ckpt = torch.load(f"ckpt_rank{dist.get_rank()}_{step}.pt", weights_only=False)
model.load_state_dict(ckpt["model"])
dataset.load_state_dict(ckpt["dataset"])
```

---

## Limitations

| Limitation | Notes |
|------------|-------|
| In-progress shard re-reads from scratch | At most one shard per worker — bounded re-processing |
| File list must not change between checkpoint and resume | Added/removed files cause `CheckpointMismatchError` |
| `num_workers` must not change | Different splits → mismatch error |
| `shuffle_seed` must not change | Different splits → mismatch error |
| Duplicate batches from in-progress shard | Up to one shard of re-delivered batches — acceptable for SGD |
