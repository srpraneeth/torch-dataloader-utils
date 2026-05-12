# Roadmap

## V2 (Released)

### Rank-Aware DDP Sharding

`TargetSizeSplitStrategy` and `RoundRobinSplitStrategy` now accept `num_ranks` and `rank` parameters. `StructuredDataset` and `IcebergDataset` expose these on both the constructor and `create_dataloader()`. Splits are assigned using interleaved slicing (`all_splits[rank::num_ranks]`) so each DDP rank reads a disjoint subset of files. Works with PyTorch DDP, Accelerate, and Horovod.

### Deterministic Distributed Shuffling

`shuffle=True` with `set_epoch(epoch)` produces a globally consistent shuffle across all ranks. The full chunk list is shuffled before rank slicing — if each rank shuffled independently, ranks could overlap. All ranks call `set_epoch(epoch)` with the same value and get the same globally shuffled assignment, with no duplicates and no missed samples. Epoch number is used as a seed offset for full reproducibility.

### Row-Group-Aware Split Scheduling

`TargetSizeSplitStrategy` reads Parquet row group metadata (byte sizes, row counts) once in the main process and packs consecutive row groups into target-sized chunks. A single large file can be split across multiple workers at row group boundaries — each worker reads only its assigned row groups via `pq.ParquetFile.read_row_groups()`. Assignment uses greedy LPT (longest-processing-time) heap scheduling for near-optimal load balance.

### Projection and Predicate Pushdown

`columns=` projects down to only the required columns before data leaves storage. `filters=` applies predicate pushdown at both the file level (via Iceberg partition pruning) and row group level (via Parquet statistics). Arrow never reads columns or row groups that the filter eliminates.

### ORC Sub-File Splitting

ORC files are now split at stripe boundaries, matching the fine-grained load balancing that Parquet has had since V1. Because PyArrow does not expose per-stripe row counts, the row count is approximated uniformly as `nrows / nstripes`. The reader calls `ORCFile.read_stripe()` for assigned stripes — true random access, no full scan.

### ORC Support for Iceberg Tables

`_detect_format` now handles ORC-backed Iceberg tables. Iceberg tables storing data in ORC format are fully supported.

### Arrow-Native Zero-Copy Batching

The entire pipeline is `Parquet / ORC → Arrow RecordBatch → tensor` — no pandas, no intermediate Python objects. `DataLoader` is constructed with `batch_size=None` so PyTorch never re-batches already-batched Arrow output. Numeric columns convert to tensors with a single memory copy; non-numeric columns (strings, timestamps) become Python lists without materializing intermediate arrays.

### Multi-Worker Integration Tests on Linux CI

The full multi-worker integration test suite now runs on Linux in CI. macOS `spawn` mode caused deadlocks with pyarrow generators; Linux `fork` mode works correctly.

---

## V2 (Pending)

### Benchmark Suite

Credibility for any data infrastructure project requires numbers. Planned benchmarks:

- **Throughput** — rows/sec and batches/sec vs. naive `IterableDataset`, `torchdata`, and WebDataset baselines
- **I/O amplification** — bytes read per training sample, across 1 / 4 / 8 workers and 1 / 4 ranks
- **Startup latency** — time from `create_dataloader()` to first batch, as a function of file count and format
- **GPU utilization** — DataLoader stall fraction under realistic S3 latency (simulated with throttled reads)
- **Shuffle overhead** — wall time cost of `set_epoch()` as split count grows

Without this, adoption arguments are anecdotal.

### Mid-Epoch Checkpoint and Resume

Persist which splits have been fully consumed so that on crash or restart the DataLoader can skip already-processed splits and resume from the partial one. Epoch number is checkpointed alongside model weights for deterministic shuffle resumption.

This will integrate with PyTorch's `state_dict()` / `load_state_dict()` protocol — the same interface used by `StatefulDataLoader` from `torchdata`. Critical for long-running jobs on preemptible / spot instances.

### Shuffle Improvements

- **Record-level shuffle** via a configurable in-memory shuffle buffer. Rows within the buffer are shuffled before yielding. Buffer size is tunable to balance randomness against memory usage.
- **Row-level interleaving** across files within a split — yield one row (or one batch) from each file in rotation rather than finishing one file before starting the next.

### Observability

Metrics exposed per worker: rows read, bytes read, worker utilization, idle time. Useful for diagnosing load imbalance and tuning `split_bytes` / `num_workers`. Planned output channels: structured log lines (default), optional Prometheus metrics.

### Delta Lake Support

Iceberg is fully supported. Delta Lake is the other dominant open table format — adding `DeltaDataset` via `delta-rs` would follow the same pattern (resolve table snapshot → data files → existing split/read pipeline) and cover the significant portion of the ecosystem that uses Databricks or Spark with Delta.

### Testing Infrastructure

- **GCS and Azure real-backend CI** — S3 (moto) covers the shared fsspec/PyFileSystem code path. V2 adds Docker Compose-based GCS (`fake-gcs-server`) and Azure (Azurite) CI tests to catch per-backend auth, path format, and `stat()` response differences.

---

## V3

### Adaptive Dynamic Splitting

Rebalance splits across workers during iteration if some workers finish significantly faster than others. Useful for heterogeneous file sizes where static LPT scheduling still leaves some workers idle near the end of an epoch.

### Memory-Aware Batching

Adaptive batch sizing based on available memory. Useful for workloads with variable row widths (e.g. embeddings with variable-length fields) where a fixed `batch_size` can trigger OOM on wide batches.

### Schema Validation and Evolution

Enforce an expected schema at `create_dataloader()` time — validate column names, types, and nullability against a user-provided schema before iteration begins. Emit a clear error rather than a cryptic Arrow type mismatch mid-epoch. Also handle schema evolution across files in the same dataset (e.g. new columns added to later partitions).

---

## Considered and Declined

### Rust Backend

A Rust rewrite of the file-reading layer (decompression, Arrow conversion, scheduling) would add enormous implementation complexity. The bottleneck in cloud training is almost always **network I/O latency**, not CPU throughput — PyArrow's internals are already C++ and are not meaningfully slower than a Rust equivalent for I/O-bound workloads. The compressor that decompresses a Snappy-encoded row group in 2 ms versus 1 ms does not move GPU utilization when the preceding S3 `GET` took 80 ms. A Rust backend is the right answer if and only if decompression or Arrow conversion shows up as a CPU bottleneck in profiling — which it has not. Revisit if benchmarks show otherwise.

### GPU-Aware Prefetch Scheduling

Dynamically adjusting prefetch depth based on GPU utilization feedback would require a tight coupling between the DataLoader and the training loop, turning a clean data-loading library into a training-loop framework. PyTorch's `DataLoader` prefetch queue already overlaps I/O with compute for the common case. Defer until there is evidence that static prefetch depth is a measurable bottleneck.

### Query-Planner-Style Execution (`loader.filter().select().batch()`)

An execution-plan interface would require building an expression tree, an optimizer, and a physical plan evaluator — essentially writing a small query engine. Arrow and Parquet already support projection and predicate pushdown natively via `columns=` and `filters=`; there is no missing capability, only a more fluent API surface. The additional abstraction layer adds complexity without improving performance or expressiveness for the target workload. Not planned.
