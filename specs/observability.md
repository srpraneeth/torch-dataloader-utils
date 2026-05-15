# Spec: Observability

## Overview

Six observability layers, each independently useful:

| Layer | When | Level | Always-on? |
|-------|------|-------|-----------|
| Startup summary | `create_dataloader()` | INFO | Yes |
| Split assignment table | `_generate_splits()` | DEBUG | Yes |
| Load balance warning | `_generate_splits()` | WARNING | Yes |
| Progress bars | During iteration, per-file | — | `show_progress=True` |
| Epoch summary | `get_metrics()` | INFO | Yes |
| Metrics collection | During iteration | — | Always |

"Always-on" means the log line is emitted whenever the logger level allows it —
no flag needed. Progress bars require `show_progress=True`.

---

## Layer 1 — Startup Summary

Logged by `create_dataloader()` at INFO level, once, before the DataLoader is
returned. Gives a complete picture of what the loader is about to do.

```
DataLoader ready
  path         : s3://bucket/train/
  format       : parquet
  files        : 50  (12.4 GB total)
  workers      : 4   (rank 0 / 1)
  batch_size   : 1024
  strategy     : TargetSizeSplitStrategy  target=128 MiB
  shuffle      : True  seed=42
  columns      : feature_a, feature_b, label  (3 of 12)
  filters      : yes
  output_fmt   : torch
```

Implemented as a single `logger.info(msg, extra={...})` call with all fields
in `extra` for JSON log formatter compatibility.

---

## Layer 2 — Split Assignment Table

Logged by `_generate_splits()` at DEBUG level, once per `set_epoch()` call.
Full breakdown of every worker's assignment — essential for debugging load
imbalance, verifying DDP sharding, or confirming row-range sub-splitting.

```
Split assignment (epoch=0, 4 workers, rank 0/1):
  Worker 0:  12 splits  |  3,201,024 rows  |  3.1 GB
    f00.parquet  rows [0, 131072)        128 MiB
    f00.parquet  rows [131072, 262144)   128 MiB
    f01.parquet  rows [0, 131072)        128 MiB
    ...
  Worker 1:  12 splits  |  3,198,720 rows  |  3.1 GB
    ...
  Worker 2:  11 splits  |  2,930,688 rows  |  2.9 GB
    ...
  Worker 3:  11 splits  |  2,928,384 rows  |  2.9 GB
    ...
  Total: 46 splits  |  12,258,816 rows  |  12.0 GB
```

Only emitted at DEBUG level — not shown in normal training runs. Enable with:
```python
logging.getLogger("torch_dataloader_utils").setLevel(logging.DEBUG)
```

---

## Layer 3 — Load Balance Warning

Logged by `_generate_splits()` at WARNING level when the largest worker
assignment exceeds 2× the smallest. Fires before training starts so the user
can act on it.

```
WARNING: Unbalanced split assignment — max worker 3.1 GB, min worker 1.4 GB
(2.2× ratio). Consider reducing target_bytes for finer-grained splits.
```

Threshold is 2.0× (hardcoded — no config knob). Only compares workers that
actually have splits (ignores empty workers when num_workers > num_files).

---

## Layer 4 — Progress Bars

### What the user sees

```
W0 | f00.parquet:  64%|████████████       | 64000/100000 [01:58<01:06, 543 rows/s]
W1 | f01.parquet:  32%|██████             | 32000/100000 [01:55<04:10, 259 rows/s]
W2 | f02.parquet:  80%|████████████████   | 80000/100000 [01:57<00:29, 682 rows/s]
```

One bar per worker per file. Each bar:
- Description: `f"W{worker_id} | {filename}"`
- `total`: `split.row_range.length` (row-range) or `split.file.record_count`
  (whole-file, from Parquet metadata) or `None` (unknown → spinner)
- `position=worker_id`, `leave=False` — bar disappears when file is done
- Updates every `progress_interval_sec` seconds via
  `mininterval=maxinterval=progress_interval_sec`

### API

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    show_progress=True,           # default False
    progress_interval_sec=120,    # default 120s
)
```

`tqdm` is imported lazily when `show_progress=True`. Raises `ImportError` with
an install hint if not installed. Added to `dev` extra in `pyproject.toml` —
not a hard runtime dependency.

---

## Layer 5 — Epoch Summary

Logged by `get_metrics()` at INFO level automatically. Aggregates all workers'
`WorkerMetrics` and logs a one-line summary plus per-worker breakdown.

```
Epoch 0 complete:  workers=4  rows=12,258,816  bytes_est=12.0GB  elapsed=142.3s  rows/s=86,148
  Worker 0:  3,201,024 rows  3.1 GB  141.2s
  Worker 1:  3,198,720 rows  3.1 GB  142.3s
  Worker 2:  2,930,688 rows  2.9 GB  139.8s
  Worker 3:  2,928,384 rows  2.9 GB  140.1s
```

`elapsed_sec` is the maximum across all workers (wall time of the epoch).
`rows/s` is total rows divided by max elapsed.

---

## Layer 6 — Metrics Collection

### `WorkerMetrics` dataclass

```python
@dataclass
class WorkerMetrics:
    worker_id: int
    rows_read: int = 0
    batches_read: int = 0
    bytes_read: int = 0       # estimated compressed bytes
    files_read: int = 0
    elapsed_sec: float = 0.0
```

### Bytes estimation

| Split type | Estimate |
|------------|----------|
| Whole-file (`row_range=None`) | `split.file.file_size` |
| Row-range, `record_count` known | `file_size × row_range.length / record_count` |
| Row-range, `record_count` unknown | `file_size` (upper bound) |

Documented as **estimated compressed bytes** in all public interfaces.

### Where counters are incremented

`read_split` accepts `metrics: WorkerMetrics | None = None` and
`pbar: tqdm | None = None`. After each batch: increments `rows_read`,
`batches_read`; calls `pbar.update(batch.num_rows)`. After each split:
increments `files_read`, `bytes_read`.

`__iter__` owns the `WorkerMetrics` for this worker. It creates it, passes it
to `read_split`, records `elapsed_sec`, then pushes to the queue.

### IPC

`StructuredDataset.__init__` allocates a `multiprocessing.Queue`. Workers
receive it via pickle (Queue wraps OS primitives — picklable in both fork and
spawn modes). Each worker pushes exactly one `WorkerMetrics` per epoch.
Queue is unbounded — workers never block.

### API

```python
dataset.get_metrics() -> list[WorkerMetrics]
```

Drains the queue, logs the epoch summary (Layer 5), and returns the list.
A second call in the same epoch returns `[]` (queue already drained).
`set_epoch()` drains and discards any stale metrics before regenerating splits.

---

## Files to Change

| File | Change |
|------|--------|
| `src/torch_dataloader_utils/observability.py` | **NEW** — `WorkerMetrics` dataclass |
| `src/torch_dataloader_utils/format/reader.py` | Accept `metrics`, `pbar` in `read_split`; increment counters; call `pbar.update()` per batch |
| `src/torch_dataloader_utils/dataset/structured.py` | Startup summary; split table + balance warning in `_generate_splits`; allocate queue; pass metrics + pbar; add `get_metrics()` / `reset_metrics()` |
| `pyproject.toml` | Add `tqdm>=4.0` to `dev` extra (already present — verify) |
| `tests/unit/test_observability.py` | **NEW** — unit tests |
| `tests/integration/test_local.py` | Integration: metrics non-zero, epoch summary logged, balance warning fires |

---

## Requirements

### Startup summary `[obs-startup]`
`create_dataloader` SHALL log an INFO summary including: path, format, file
count, total size, num_workers, rank/num_ranks, batch_size, strategy name,
shuffle state, columns (if projected), filter presence, output_format.

### Split table `[obs-splits]`
`_generate_splits` SHALL log at DEBUG level: per-worker file list with row
ranges and estimated sizes, and a total line. SHALL include epoch number.

### Balance warning `[obs-balance]`
`_generate_splits` SHALL emit a WARNING when `max_worker_bytes / min_worker_bytes > 2.0`
(ignoring workers with zero splits). Message SHALL include the ratio and a
suggestion to reduce `target_bytes`.

### Progress bars `[obs-progress]`
`create_dataloader` and `StructuredDataset.__init__` SHALL accept
`show_progress: bool = False` and `progress_interval_sec: float = 120`.
When `True`, one tqdm bar per worker per file, `position=worker_id`,
`leave=False`, updating every `progress_interval_sec` seconds.
When tqdm is not installed and `show_progress=True`, SHALL raise `ImportError`.

### Epoch summary `[obs-epoch]`
`get_metrics()` SHALL log an INFO epoch summary: total rows, total bytes,
wall-time elapsed, rows/sec, per-worker breakdown.

### Metrics `[obs-metrics]`
`WorkerMetrics` SHALL track `rows_read`, `batches_read`, `bytes_read`,
`files_read`, `elapsed_sec`. `get_metrics()` SHALL drain the queue and return
`list[WorkerMetrics]`. `set_epoch()` SHALL discard stale metrics.

### Tests `[obs-tests]`
Unit tests SHALL verify:
- `WorkerMetrics` increments correctly via `read_split`
- `get_metrics()` returns correct data after `num_workers=0` iteration
- `get_metrics()` returns `[]` before any iteration and on second drain
- Balance warning fires when max/min ratio exceeds 2.0 (mocked shards)
- Balance warning absent when splits are balanced

Integration tests SHALL verify:
- `sum(m.rows_read) == total_rows` after one epoch
- `sum(m.bytes_read) > 0` and `m.elapsed_sec > 0`
- `show_progress=True` with `num_workers=0` completes without error
- Startup summary is logged (caplog)
- Split table is logged at DEBUG (caplog)

---

## Out of Scope

- **Prometheus export**: `WorkerMetrics` is the integration point for a later adapter.
- **Idle / utilization time**: requires per-`yield` timing — deferred.
- **Combined "all workers" live bar**: each worker owns its bar independently.
