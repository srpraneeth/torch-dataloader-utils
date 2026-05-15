# Observability

Both `StructuredDataset` and `IcebergDataset` emit structured log output at every stage of the pipeline using Python's standard `logging` module.

Enable it with:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
```

---

## Startup Summary

Logged at **INFO** when `create_dataloader()` is called, before any I/O begins.

=== "StructuredDataset"
    ```
    INFO DataLoader ready
      path         : s3://bucket/train/
      format       : parquet
      files        : 50  (12.4 GB total)
      workers      : 4   (rank 0 / 1)
      batch_size   : 1024
      strategy     : TargetSizeSplitStrategy
      shuffle      : True  seed=42
      columns      : feature_a, feature_b, label
      filters      : yes
      output_fmt   : torch
    ```

=== "IcebergDataset"
    ```
    INFO IcebergDataset ready
      table        : my_db.my_table
      workers      : 4   (rank 0 / 1)
      batch_size   : 1024
      strategy     : TargetSizeSplitStrategy
      shuffle      : True  seed=42
      snapshot_id  : current
      columns      : feature_a, feature_b, label
      filters      : yes
      output_fmt   : torch
    ```

---

## Split Assignment

Logged at **INFO** before any worker starts reading. Shows exactly which files (and row ranges) each worker is assigned. Re-logged on every `set_epoch()` call.

```
INFO Split assignment (epoch=0, 4 workers, rank 0/1):
  Worker 0:  12 splits  |  3,201,024 rows  |  3.1 GB
    f00.parquet  rows [0, 131,072)        128.0 MB
    f00.parquet  rows [131,072, 262,144)  128.0 MB
    f01.parquet  full file                 95.2 MB
    ...
  Total: 46 splits  |  12,258,816 rows  |  12.0 GB
```

!!! tip "Debugging DDP sharding"
    With `num_ranks` set, each rank logs only its own slice — verify rank 0 and rank 1 received disjoint files.

---

## Load Balance Warning

Logged at **WARNING** when the largest worker assignment exceeds **2×** the smallest:

```
WARNING Unbalanced split assignment — max worker 3.1 GB, min worker 1.4 GB
(2.2× ratio). Consider reducing target_bytes for finer-grained splits.
```

Fix with a smaller `split_bytes`:

```python
loader, dataset = StructuredDataset.create_dataloader(
    ...,
    split_bytes="64MiB",
)
```

---

## Progress Bars

Enable per-worker tqdm bars with `show_progress=True`. One bar per worker, refreshing every `progress_interval_sec` seconds (default 120s).

```python
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    show_progress=True,
    progress_interval_sec=30,
)
```

```
W0 | f00.parquet:  64%|████████████       | 64000/100000 [01:58<01:06, 543 rows/s]
W1 | f01.parquet:  32%|██████             | 32000/100000 [01:55<04:10, 259 rows/s]
```

Percentage is shown when the row count is known (Parquet). CSV/JSON shows a spinner. Requires `tqdm` (`pip install tqdm`).

Both `StructuredDataset` and `IcebergDataset` accept `show_progress` and `progress_interval_sec`.

---

## Per-File Logs

Logged at **INFO** by each worker after every file (or sub-file split) completes — no flag needed, works in any log setup:

```
INFO Worker 0 file done: f00.parquet  rows=131,072  batches=128  bytes_est=128.0 MB  elapsed=1.842s
```

---

## Epoch Summary and Metrics

Call `dataset.get_metrics()` after the loop to log an epoch summary and get per-worker counters:

```python
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for batch in loader:
        train(batch)

    for m in dataset.get_metrics():
        print(f"worker={m.worker_id}  rows={m.rows_read:,}  bytes={m.bytes_read / 1e9:.2f} GB")
```

Logged automatically at **INFO**:

```
INFO Epoch 0 complete:  workers=4  rows=12,258,816  bytes_est=12.0 GB  elapsed=142.3s  rows/s=86,148
  Worker 0:  3,201,024 rows  3.1 GB  141.2s
  Worker 1:  3,198,720 rows  3.1 GB  142.3s
```

**`WorkerMetrics` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `worker_id` | `int` | Worker index (0-based) |
| `rows_read` | `int` | Total rows yielded |
| `batches_read` | `int` | Total RecordBatches yielded |
| `bytes_read` | `int` | Estimated compressed bytes read |
| `files_read` | `int` | Number of splits processed |
| `elapsed_sec` | `float` | Wall time from first to last batch |

`get_metrics()` drains the internal queue — calling it a second time returns `[]`. `set_epoch()` automatically discards stale metrics from the previous epoch.

---

## JSON Logging

All log lines carry structured `extra` fields. With a JSON formatter (e.g. [`python-json-logger`](https://github.com/madzak/python-json-logger)) you get machine-readable records automatically:

```python
from pythonjsonlogger import jsonlogger
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logging.getLogger("torch_dataloader_utils").addHandler(handler)
```

Events and their fields:

| Event | Fields |
|-------|--------|
| `file_done` | `worker_id`, `file`, `rows_read`, `batches_read`, `bytes_read` |
| `shard_done` | `worker_id`, `files_read`, `rows_read`, `batches_read`, `bytes_read`, `elapsed_sec` |
| `epoch_done` | `epoch`, `workers`, `total_rows`, `total_bytes`, `elapsed_sec`, `rows_per_sec` |
