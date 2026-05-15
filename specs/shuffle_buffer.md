# Spec: Record-Level Shuffle Buffer

## Problem

The existing `shuffle=True` parameter operates at the **chunk level** — before split
assignment, the list of file-split chunks is reordered using `seed + epoch`. Within a chunk,
rows are read in file order. If a Parquet file is sorted by timestamp or user ID, a worker
reads all its assigned rows in that sorted order. Consecutive training batches will be
temporally correlated even when chunk-level shuffle is enabled.

Record-level shuffle draws rows from a pool of buffered records before yielding each batch —
the standard approach in tf.data, HuggingFace datasets, and most production training
pipelines.

---

## Two Shuffle Levels — Complementary, Independent

| Level | Parameter | Scope | When to use |
|-------|-----------|-------|-------------|
| Chunk | `shuffle=True` | Reorders file-split chunks before assignment | Break correlation between epochs |
| Record | `shuffle_buffer_size=N` | Rows within each worker's stream | Break correlation within a chunk |

They are independent axes. Either can be used alone or together:

```
shuffle=True, shuffle_buffer_size=None   → chunk order randomised, rows in file order
shuffle=True, shuffle_buffer_size=50000  → chunk order randomised + rows mixed across chunks
shuffle=False, shuffle_buffer_size=50000 → fixed chunk order, rows mixed within each chunk
```

Best practice for training: use both. Chunk shuffle ensures different epochs see different
file orderings; the record buffer mixes rows across chunks so each batch samples broadly
from the worker's assigned data.

---

## Buffer Location — In-Memory, Per Worker Process

The buffer lives entirely inside each **worker process's heap**. It never crosses a process
boundary. IPC only happens once per output batch — when a completed shuffled RecordBatch is
put into the DataLoader's output pipe.

```
Worker 0 process (its own memory space)
  ┌──────────────────────────────────────────┐
  │  Arrow Table buffer (N rows) ← heap RAM  │
  │  RNG state                               │
  │  _iter_shard generator                   │
  └──────────────────────────────────────────┘
           ↓  yield shuffled RecordBatch
  DataLoader output pipe  ← IPC happens here, once per batch
           ↓
  Main process training loop
```

With `num_workers=8` there are **8 independent buffers**, one per worker. No coordination,
no locks. Total memory = `num_workers × shuffle_buffer_size × row_width`.

For `num_workers=0` (single process), the buffer lives in the main process heap — no IPC
at all.

---

## Algorithm — Reservoir-Style Streaming Buffer

The buffer holds individual rows (not batches). It is maintained as an Arrow Table for
efficient random access via `table.take(indices)`.

```
1. Read from _iter_shard until buffer has >= shuffle_buffer_size rows
2. While source still has rows:
     a. Pick batch_size random indices from the buffer (without replacement)
     b. Yield table.take(sorted_indices) as one output RecordBatch
     c. Remove those rows from the buffer
     d. Refill buffer from source until buffer has >= shuffle_buffer_size rows again
        (or source is exhausted)
3. Drain: shuffle the remaining buffer rows with Fisher-Yates, yield in batch_size chunks
```

Sorting picked indices before `take()` is a cache-efficiency detail — Arrow reads
column chunks in order so sorted indices avoid seeking backwards.

### Why not true Fisher-Yates streaming?

Classic streaming shuffle picks one random row, replaces it, picks the next. That produces
O(batch_size) `take()` calls per output batch. Batch-draining (pick `batch_size` rows at
once) amortises the index manipulation cost with no loss in shuffle quality for training.

### Shuffle quality

With `shuffle_buffer_size >= dataset_size / num_workers`, the buffer covers the entire
worker's assignment — equivalent to a full in-memory shuffle of that worker's data.
Smaller buffers trade quality for memory: a buffer of 50,000 rows means each output
row is drawn uniformly from a 50,000-row window of the input stream.

---

## New Parameter

```python
shuffle_buffer_size: int | None = None
```

Added to `StructuredDataset.__init__`, `StructuredDataset.create_dataloader`,
`IcebergDataset.__init__`, and `IcebergDataset.create_dataloader`.

| Value | Behaviour |
|-------|-----------|
| `None` | No record-level shuffle — existing behaviour, fully backward compatible |
| `0` | Same as `None` |
| `N > 0` | Record-level shuffle with a buffer of N rows |

`shuffle_buffer_size` is stored on the dataset and passed to `BaseDataset` via
`_init_splits_and_observability`. No change to `SplitStrategy` or `read_split`.

---

## RNG Seeding

Each worker gets a deterministic per-epoch RNG:

```python
rng_seed = self._shuffle_seed * 100_000 + self._epoch * 1_000 + worker_id
rng = np.random.default_rng(rng_seed)
```

- Same `shuffle_seed` + same `epoch` + same `worker_id` → same shuffle order
- Different epochs → different within-epoch row order
- Different workers → different shuffles (workers don't shuffle their data identically)

`numpy.random.default_rng` (PCG64) is used rather than Python's `random` — it's faster
for large index arrays and produces higher quality randomness.

---

## Implementation — BaseDataset

### _init_splits_and_observability

Add `shuffle_buffer_size` field:

```python
self._shuffle_buffer_size: int = max(0, shuffle_buffer_size or 0)
```

### __iter__ — wrap _iter_shard output

```python
source = self._iter_shard(shard, worker_id, metrics, pbar)
if self._shuffle_buffer_size > 0:
    rng_seed = self._shuffle_seed * 100_000 + self._epoch * 1_000 + worker_id
    rng = np.random.default_rng(rng_seed)
    yield from _shuffle_buffer_iter(source, self._shuffle_buffer_size, self._batch_size, rng)
else:
    yield from source
```

### _shuffle_buffer_iter — standalone generator function

Defined at module level in `base.py` (not a method — no `self` needed):

```python
def _shuffle_buffer_iter(
    source: Iterator[Any],
    buffer_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[Any]:
    import pyarrow as pa
    import numpy as np

    buffer: pa.Table | None = None

    def _append(table: pa.Table | None, batch) -> pa.Table:
        t = pa.Table.from_batches([batch]) if not isinstance(batch, pa.Table) else batch
        return pa.concat_tables([table, t]) if table is not None else t

    def _drain_batch(table: pa.Table) -> tuple[Any, pa.Table]:
        n = min(batch_size, len(table))
        idx = np.sort(rng.choice(len(table), size=n, replace=False))
        out = table.take(idx)
        mask = np.ones(len(table), dtype=bool)
        mask[idx] = False
        remaining = table.filter(pa.array(mask))
        return out.to_batches()[0], remaining

    # Fill initial buffer
    for batch in source:
        buffer = _append(buffer, batch)
        while buffer is not None and len(buffer) >= buffer_size:
            out, buffer = _drain_batch(buffer)
            yield out
            # Refill: pull one more batch to keep buffer topped up
            try:
                next_batch = next(source)  # NOTE: source is an iterator
                buffer = _append(buffer, next_batch)
            except StopIteration:
                break

    # Drain remainder with full shuffle
    if buffer is not None and len(buffer) > 0:
        shuffled_idx = rng.permutation(len(buffer))
        for i in range(0, len(buffer), batch_size):
            chunk = np.sort(shuffled_idx[i : i + batch_size])
            yield buffer.take(chunk).to_batches()[0]
```

> **Note on `source` as iterator**: `_iter_shard` is a generator, so converting it to an
> iterator with `iter()` is not needed — generators are already iterators. The `for batch in
> source` loop and `next(source)` calls can be mixed safely.

---

## Memory

| `shuffle_buffer_size` | 20 float32 cols | 100 float32 cols |
|-----------------------|-----------------|------------------|
| 10,000 rows | 800 KB | 4.0 MB |
| 50,000 rows | 4.0 MB | 20 MB |
| 100,000 rows | 8.0 MB | 40 MB |
| 500,000 rows | 40 MB | 200 MB |

Memory is per worker. With `num_workers=8` and a 50,000-row buffer of 100 columns:
8 × 20 MB = 160 MB total. Reasonable for a training machine.

Document recommended values:
- **10,000** — low memory, moderate shuffle quality
- **50,000** — good default for most training workloads
- **`dataset_size / num_workers`** — full shuffle quality, equivalent to in-memory shuffle

---

## Output Format Interaction

`_shuffle_buffer_iter` operates on Arrow RecordBatches and yields Arrow RecordBatches.
The existing `convert_batch(batch, output_format)` call in `StructuredDataset._iter_shard`
runs BEFORE the buffer — it converts to torch/numpy/dict. This means the buffer would
receive already-converted tensors, not Arrow batches.

**Fix**: move `convert_batch` to AFTER the shuffle buffer. In `StructuredDataset._iter_shard`,
yield raw Arrow RecordBatches; in `__iter__`, apply `convert_batch` after the buffer:

```python
# __iter__ in BaseDataset
source = self._iter_shard(shard, worker_id, metrics, pbar)  # yields raw RecordBatches
if self._shuffle_buffer_size > 0:
    rng = np.random.default_rng(...)
    source = _shuffle_buffer_iter(source, self._shuffle_buffer_size, self._batch_size, rng)
for batch in source:
    yield self._convert_output(batch)   # new method on BaseDataset
```

`_convert_output` is a new abstract or concrete method:
- `StructuredDataset`: calls `convert_batch(batch, self._output_format)`
- `IcebergDataset`: same

This requires `_iter_shard` to yield raw Arrow RecordBatches, with conversion lifted to
`__iter__`. Both current `_iter_shard` implementations yield converted output — this needs
to change.

---

## Metrics

`metrics` (rows_read, batches_read, bytes_read, files_read) are accumulated inside
`_iter_shard` as data is read from disk. The shuffle buffer is applied after `_iter_shard`
yields — metrics track I/O, not the post-shuffle output. This is the correct semantics and
requires no changes to metrics.

The number of output batches may differ from `metrics.batches_read` when `shuffle_buffer_size`
is set (because the buffer recomposes rows into new batches). This is expected and fine —
`batches_read` counts disk reads, not yielded batches.

---

## Files to Change

| File | Change |
|------|--------|
| `src/torch_dataloader_utils/dataset/base.py` | Add `_shuffle_buffer_size`, `_shuffle_seed` to `_init_splits_and_observability`; add `_shuffle_buffer_iter` generator; update `__iter__` to wrap source; add `_convert_output` method |
| `src/torch_dataloader_utils/dataset/structured.py` | Add `shuffle_buffer_size` param; `_iter_shard` yields raw RecordBatches (conversion moved to base) |
| `src/torch_dataloader_utils/dataset/iceberg.py` | Add `shuffle_buffer_size` param; same conversion lift |
| `tests/unit/test_shuffle_buffer.py` | **NEW** — unit tests |
| `docs/` | Update splits.md or add shuffle section to observability.md |

---

## Key Design Decisions

**Row-level, not batch-level**: Row-level gives true record shuffle. Batch-level only
reorders batches — rows within a batch (from the same row group) remain correlated.

**Arrow Table as buffer**: `table.take(sorted_indices)` is a zero-copy column-oriented
operation. Much faster than decomposing batches into Python dicts or lists of rows.

**Conversion after buffer**: The buffer must operate on Arrow data so it can use `take()`.
Converting to torch tensors before buffering would require tensor indexing instead, which
works but loses the schema-aware Arrow operations. Moving conversion to `__iter__` after the
buffer is a clean separation: storage format → shuffle → output format.

**Independent of chunk shuffle**: Decoupled so users can mix freely. Chunk shuffle is about
epoch-to-epoch diversity; record buffer is about within-epoch batch diversity.

**`shuffle_buffer_size=None` default**: Fully backward compatible — existing code with
`shuffle=True` continues to work exactly as before with only chunk-level shuffle.

---

## Verification

```bash
.venv/bin/python -m pytest tests/unit/test_shuffle_buffer.py -v
.venv/bin/python -m pytest tests/unit/ -v
```

**Key test scenarios:**

1. `shuffle_buffer_size=None` — output identical to current (no record shuffle)
2. All rows present, no duplicates after buffer
3. Output row order differs from input order (shuffle actually happened)
4. Deterministic: same seed + epoch + worker_id → same output order
5. Different epochs → different output order
6. Buffer smaller than dataset — works correctly (partial buffer fill, drain)
7. Buffer larger than dataset — drains entirely in final step
8. Last batch may be smaller than `batch_size` (tail handling)
9. Empty shard — yields nothing without error
