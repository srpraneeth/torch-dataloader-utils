# Limitations

## DDP / Multi-GPU Rank Sharding

In V1, each DDP rank runs its own `DataLoader` that reads the **full dataset**. `accelerator.prepare()` wraps the DataLoader but does not re-partition the underlying files across ranks.

```
V1 behavior:
  Rank 0: DataLoader → reads ALL files → trains on its half of batches
  Rank 1: DataLoader → reads ALL files → trains on its half of batches
  → full dataset I/O per rank, I/O cost scales with rank count
```

For true rank-level file partitioning in V1, use the `split_strategy` escape hatch to build separate `StructuredDataset` instances per rank with disjoint file assignments.

V2 will add a native `accelerator` parameter to `create_dataloader()` that handles this automatically.

## No Record-Level Shuffle

Shuffle operates at the **chunk level** (row group or file granularity), not at the individual row level. Within a chunk, rows are always read in their storage order.

This is intentional: record-level shuffle would require reading an entire row group into memory to shuffle it, negating the streaming benefit. Chunk-level shuffle provides sufficient randomness for most training workloads where row ordering within a file is not systematically biased.

If record-level shuffle is required, pre-shuffle your data files before training.

## ORC: No Sub-File Splitting

ORC files are treated as a single unsplittable chunk per file. Multiple ORC files are still load-balanced across workers by file size using LPT scheduling — the limitation is only that a **single large ORC file cannot be split across workers**.

ORC does have stripes with footer metadata (row counts, byte offsets) — sub-file splitting is technically feasible and planned for V2. In V1, for good parallelism, partition large ORC files into smaller ones (128–512 MiB each) so the scheduler has enough chunks to balance.

## CSV / JSON / JSONL: No Sub-File Splitting

Text formats have no footer metadata and no seek-friendly internal structure. Multiple files are still load-balanced across workers by file size using LPT scheduling — the limitation is only that a **single large file cannot be split across workers**.

If you have a few very large CSV or JSONL files, split them into smaller files (128–512 MiB each) to give the strategy enough chunks to distribute evenly.

## Iceberg: Equality Deletes Not Supported

Tables with **equality delete files** (produced by some engines' `DELETE` or `MERGE INTO` operations) will raise `NotImplementedError` from `pyiceberg` — equality deletes are not implemented in the pyiceberg Arrow reader.

**Workaround:** compact the table before training (`ALTER TABLE ... REWRITE DATA FILES`) to produce clean data files with no outstanding deletes.

Position delete files are supported — deleted rows are automatically excluded.

## Iceberg: Sub-File Splitting Disabled with Delete Files

When position delete files are present, `IcebergDataset` reads each data file whole (no row group splitting). Position delete offsets reference absolute row positions in the original file, so splitting at row group boundaries would require reconstructing the delete mask per chunk — not implemented in V1.

## `arrow` / `dict` Output: Explicit `collate_fn` Required

`output_format="arrow"` and `output_format="dict"` always require a `collate_fn` to be passed explicitly. PyTorch's default collate cannot handle `pyarrow.RecordBatch` or arbitrary dicts.

This is enforced at **construction time** — you get a `ValueError` immediately rather than a cryptic error during iteration.

```python
loader, _ = StructuredDataset.create_dataloader(
    ...,
    output_format="arrow",
    collate_fn=lambda x: x,   # required
)
```
