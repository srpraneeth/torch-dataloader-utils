# Spec: Mid-Epoch Checkpoint and Resume

## Problem

Training jobs that run for hours or days crash. When they do there is no way to resume the
DataLoader from where it left off — the next run restarts the epoch from scratch.

The common workaround is to checkpoint `(epoch, step)` and on resume iterate through the
DataLoader discarding batches until `step` is reached. This is suboptimal: all data up to
the checkpoint is re-read and thrown away — the same I/O cost as starting over.

The target behaviour: save `dataset.state_dict()` alongside `model.state_dict()`. On resume
call `dataset.load_state_dict(state)` and completed shards are skipped with **zero I/O** —
workers early-return before touching the filesystem.

---

## Checkpoint Granularity — Shard Level

The natural checkpoint unit is a **shard**: the full set of file splits assigned to one
worker for one epoch. A shard maps 1:1 to a worker ID and is the unit workers already
process end-to-end.

| Granularity | Re-processing on resume | I/O on resume |
|-------------|------------------------|---------------|
| Step (current pattern) | None | Full re-read up to checkpoint step |
| **Shard (chosen)** | ≤ 1 shard re-reads from start | Zero for completed shards |
| Batch | Near zero | Same as shard — reading to discard |

Batch-level skipping only skips the yield, not the I/O. Shard-level is the honest boundary.

**Re-processing bound:** the in-progress shard at crash time re-reads from its start. With
`num_workers=8` and the default 128 MiB split target this is at most 12.5% of one epoch's
data. Completed shards cost nothing.

---

## State Representation

The state stores **shard content** — the actual file paths and row ranges — not worker IDs.
Worker IDs are only valid if `num_workers`, `shuffle_seed`, and the file list are all
identical on resume. Storing content makes validation explicit and failure loud.

```python
{
    "epoch": 3,
    "_num_workers": 8,      # stored for validation / error messages only
    "_shuffle_seed": 42,    # stored for validation / error messages only
    "completed_shards": [
        {
            "splits": [
                {"path": "s3://bucket/part-0001.parquet", "row_offset": 0,    "row_length": 250000},
                {"path": "s3://bucket/part-0002.parquet", "row_offset": None, "row_length": None},
            ]
        },
        {
            "splits": [
                {"path": "s3://bucket/part-0003.parquet", "row_offset": 0, "row_length": 128000},
            ]
        },
    ]
}
```

`row_offset=None, row_length=None` means the whole file (no sub-file split).

---

## API

### state_dict()

Call at any point during training — typically every N steps alongside `model.state_dict()`.

```python
torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "dataset":   dataset.state_dict(),
    "epoch":     epoch,
    "step":      step,
}, f"checkpoint_{epoch}_{step}.pt")
```

Returns only shards that have been **fully completed** as of the call time. The in-progress
shard is not included — it will re-read from scratch on resume.

### load_state_dict()

Call after constructing the dataset with identical parameters, before the DataLoader
iterates for the resumed epoch. Replaces `set_epoch()` for the resumed epoch.

```python
ckpt = torch.load("checkpoint.pt")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
dataset.load_state_dict(ckpt["dataset"])   # validates + restores state

for batch in loader:   # completed shards yield nothing — zero I/O
    train(batch)
```

### Full training loop pattern

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    shuffle=True,
    shuffle_seed=42,
    num_workers=8,
)

start_epoch = 0

if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    dataset.load_state_dict(ckpt["dataset"])   # raises if params changed
    start_epoch = ckpt["epoch"]

for epoch in range(start_epoch, num_epochs):
    if epoch != start_epoch:
        dataset.set_epoch(epoch)               # clears completed shards for new epoch

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

## Validation on load_state_dict()

`load_state_dict` must fail loudly if the stored state cannot be safely applied to the
current dataset. Silent mismatches would corrupt training without any visible error.

### Validation steps

```
1. Restore _epoch from state
2. Call _generate_splits() to regenerate splits for that epoch
3. For each completed_shard in state["completed_shards"]:
       find a shard in the regenerated splits whose splits list matches exactly
           (same file paths, same row_offset, same row_length for every FileSplit)
       if no match found → raise CheckpointMismatchError
4. If all completed shards matched → record the matched worker IDs as _completed_workers
5. Log: "Resumed from checkpoint: epoch=3  completed_workers=[0, 1, 3, 5]  skipped=4/8 shards"
```

Matching is by **content**, not by worker ID. This means if `num_workers` changes but the
splits happen to produce the same content (unlikely but possible), resume still works
correctly. If content differs for any reason, it fails.

### CheckpointMismatchError

```
CheckpointMismatchError: Checkpoint shard does not match any current split.

  Checkpoint shard splits:
    part-0001.parquet  rows [0, 250,000)
    part-0002.parquet  full file

  No matching shard found in regenerated splits for epoch=3.

  Checkpoint was saved with: num_workers=8  shuffle_seed=42
  Current dataset has:       num_workers=4  shuffle_seed=42

  Likely cause: num_workers changed between checkpoint and resume.
  Reconstruct the dataset with num_workers=8 or discard this checkpoint.
```

---

## How Workers Signal Completion

Workers already push a `WorkerMetrics` object to `_metrics_queue` (or `_metrics_local`) at
the end of `BaseDataset.__iter__`'s `finally` block. This is the shard-complete signal.

`state_dict()` drains this queue to find completed workers, then looks up their shard
content from `self._splits`:

```python
def state_dict(self) -> dict:
    self._drain_to_completed()    # drain queue → update _completed_workers
    completed_shards = []
    for worker_id in sorted(self._completed_workers):
        shard = self._splits[worker_id]
        completed_shards.append({
            "splits": [
                {
                    "path":       fs.file.path,
                    "row_offset": fs.row_range.offset if fs.row_range else None,
                    "row_length": fs.row_range.length if fs.row_range else None,
                }
                for fs in shard.splits
            ]
        })
    return {
        "epoch":             self._epoch,
        "_num_workers":      self._num_workers,
        "_shuffle_seed":     getattr(self, "_shuffle_seed", None),
        "completed_shards":  completed_shards,
    }
```

---

## BaseDataset Changes

### New fields

```python
self._completed_workers: set[int] = set()   # worker IDs whose shards are done
```

### __iter__ — early return for completed workers

```python
def __iter__(self):
    info = get_worker_info()
    worker_id = info.id if info is not None else 0
    is_main_process = info is None

    if worker_id in self._completed_workers:
        logger.debug("Worker %d: shard already completed — skipping (resume)", worker_id)
        return

    # ... rest unchanged
```

### state_dict()

```python
def state_dict(self) -> dict:
    self._drain_to_completed()
    completed_shards = [...]   # as above
    return {
        "epoch":            self._epoch,
        "_num_workers":     self._num_workers,
        "_shuffle_seed":    getattr(self, "_shuffle_seed", None),
        "completed_shards": completed_shards,
    }
```

### load_state_dict()

```python
def load_state_dict(self, state: dict) -> None:
    self.reset_metrics()
    self._completed_workers = set()

    saved_epoch = state["epoch"]
    self._epoch = saved_epoch
    self._splits = self._generate_splits()   # regenerate for saved epoch

    for shard_state in state["completed_shards"]:
        matched_worker_id = self._match_shard(shard_state, state)
        self._completed_workers.add(matched_worker_id)

    logger.info(
        "Resumed from checkpoint: epoch=%d  completed=%d/%d shards",
        self._epoch,
        len(self._completed_workers),
        len(self._splits),
    )
```

### _match_shard() — content-based matching

```python
def _match_shard(self, shard_state: dict, full_state: dict) -> int:
    target = shard_state["splits"]
    for shard in self._splits:
        candidate = [
            {
                "path":       fs.file.path,
                "row_offset": fs.row_range.offset if fs.row_range else None,
                "row_length": fs.row_range.length if fs.row_range else None,
            }
            for fs in shard.splits
        ]
        if candidate == target:
            return shard.id
    raise CheckpointMismatchError(shard_state, full_state, self)
```

### set_epoch() — clears completed workers

```python
def set_epoch(self, epoch: int) -> None:
    self._completed_workers = set()   # epoch boundary — all shards restart
    # ... rest unchanged
```

### _drain_to_completed() — internal helper

```python
def _drain_to_completed(self) -> None:
    for m in self._metrics_local:
        self._completed_workers.add(m.worker_id)
    while True:
        try:
            m = self._metrics_queue.get_nowait()
            self._metrics_local.append(m)      # keep for get_metrics()
            self._completed_workers.add(m.worker_id)
        except queue.Empty:
            break
```

---

## DDP / Multi-Rank

Each rank has its own `dataset` instance. Each rank saves and loads its own state
independently — rank-aware sharding already isolates data per rank.

```python
# Save — each rank saves its own state
torch.save({
    "dataset": dataset.state_dict(),
}, f"ckpt_rank{dist.get_rank()}_{step}.pt")

# Resume — each rank loads its own state
dataset.load_state_dict(
    torch.load(f"ckpt_rank{dist.get_rank()}_{step}.pt")["dataset"]
)
```

---

## Limitations

| Limitation | Notes |
|------------|-------|
| In-progress shard re-reads from scratch | At most 1 shard per worker — bounded re-processing |
| File list must not change between crash and resume | Added/removed files cause `CheckpointMismatchError` |
| `num_workers` must not change | Different num_workers → different splits → mismatch error |
| `shuffle_seed` must not change | Different seed → different splits → mismatch error |
| `load_state_dict` replaces `set_epoch` for the resumed epoch | Calling both clears `_completed_workers` |
| Duplicate batches from in-progress shard | Up to 1 shard of re-delivered batches — acceptable for SGD |

---

## Files to Change

| File | Change |
|------|--------|
| `src/torch_dataloader_utils/dataset/base.py` | Add `_completed_workers`, `state_dict()`, `load_state_dict()`, `_match_shard()`, `_drain_to_completed()`; update `__iter__` and `set_epoch()` |
| `src/torch_dataloader_utils/observability.py` | Export `CheckpointMismatchError` |
| `tests/unit/test_checkpoint.py` | **NEW** — state_dict round-trip, load validates content, mismatch raises, set_epoch clears state |
| `tests/integration/test_local.py` | Add: run partial epoch, checkpoint, resume, verify no missing rows and no duplicate rows beyond one shard |
| `docs/observability.md` | Add checkpoint/resume section |

No changes needed to `StructuredDataset`, `IcebergDataset`, split strategies, or the reader.
The feature lives entirely in `BaseDataset`.

---

## Verification

```bash
# Unit tests
.venv/bin/python -m pytest tests/unit/test_checkpoint.py -v

# Integration
.venv/bin/python -m pytest tests/integration/test_local.py -k checkpoint -v
```

**Key test scenarios:**

1. `state_dict` round-trip — save and reload, verify epoch and shard content match
2. Resume skips completed shards — simulate 3 of 4 workers done, resume, verify only 1 shard re-read
3. Mismatch raises — change `num_workers` between save and load, verify `CheckpointMismatchError`
4. File list change raises — remove a file, verify `CheckpointMismatchError`
5. `set_epoch` clears state — call `set_epoch` after `load_state_dict`, verify `_completed_workers` is empty
6. Full integration — partial epoch, crash simulation, resume, assert `total_rows == dataset_rows` with at most one shard of duplicates
