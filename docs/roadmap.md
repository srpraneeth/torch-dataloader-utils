# Roadmap

V1 is a stable, production-usable foundation. V2 focuses on distributed training ergonomics, fault tolerance, expanded format support, and observability.

## V2

### Distributed Training

**Accelerate-native rank-aware split assignment** — `create_dataloader()` will accept an `accelerator` parameter. When provided, the split strategy automatically partitions files across DDP ranks so each rank reads only its assigned files, with no cross-rank data overlap.

```python
# V2 — planned API
from accelerate import Accelerator

accelerator = Accelerator()
loader, dataset = StructuredDataset.create_dataloader(
    path="s3://bucket/data/",
    format="parquet",
    num_workers=4,
    accelerator=accelerator,   # rank-aware sharding — each rank reads disjoint files
)
```

**Horovod and DeepSpeed support** — the same rank-aware mechanism extended to Horovod and DeepSpeed distributed backends.

### Mid-Epoch Checkpoint and Resume

Persist which splits have been fully consumed so that on crash or restart the DataLoader can skip already-processed splits and resume from the partial one. Epoch number is checkpointed alongside model weights for deterministic shuffle resumption.

This will integrate with PyTorch's `state_dict()` / `load_state_dict()` protocol — the same interface used by `StatefulDataLoader` from `torchdata`.

### Shuffle Improvements

- **Record-level shuffle** via a configurable in-memory shuffle buffer. Rows within the buffer are shuffled before yielding. Buffer size is tunable to balance randomness against memory usage.
- **Row-level interleaving** across files within a split — yield one row (or one batch) from each file in rotation rather than finishing one file before starting the next.

### ORC Sub-File Splitting

ORC files have **stripes** with row counts and byte offsets in the file footer — equivalent to Parquet row groups. V2 will read ORC stripe metadata in the main process and generate sub-file `Split` objects at stripe boundaries, matching the fine-grained load balancing that Parquet has today.

### ORC Support for Iceberg Tables

Iceberg tables can store data in ORC format in addition to Parquet. V2 adds first-class support for ORC-backed Iceberg tables.

### Observability

Metrics exposed per worker: rows read, bytes read, worker utilization, idle time. Useful for diagnosing load imbalance and tuning `split_bytes` / `num_workers`.

### Testing Infrastructure

- **GCS and Azure real-backend CI** — S3 (moto) covers the shared fsspec/PyFileSystem code path in V1. V2 adds Docker Compose-based GCS (`fake-gcs-server`) and Azure (Azurite) CI tests to catch per-backend auth, path format, and `stat()` response differences.
- **Multi-worker DataLoader integration tests on Linux CI** — macOS `spawn` mode causes deadlocks with pyarrow generators; Linux `fork` mode works correctly. V2 CI runs the full multi-worker integration suite on Linux.

---

## V3

**Adaptive dynamic splitting** — rebalance splits across workers during iteration if some workers finish significantly faster than others. Useful for heterogeneous file sizes where static LPT scheduling still leaves some workers idle near the end of an epoch.
