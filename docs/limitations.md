# Limitations

## No Record-Level Shuffle

Shuffle operates at the **chunk level** (row group or file granularity), not at the individual row level. Within a chunk, rows are always read in their storage order.

This is intentional: record-level shuffle would require reading an entire row group into memory to shuffle it, negating the streaming benefit. Chunk-level shuffle provides sufficient randomness for most training workloads where row ordering within a file is not systematically biased.

If record-level shuffle is required, pre-shuffle your data files before training.

## ORC: Approximate Row Counts in Sub-File Splitting

ORC files are split at stripe boundaries. Because PyArrow does not expose per-stripe row counts in the file footer, row counts are **approximated uniformly** as `total_rows / num_stripes`. This means LPT scheduling for ORC is based on approximate row counts rather than exact ones.

In practice, stripes within a single ORC file tend to be similar in size, so the approximation is close. If your ORC files have very unequal stripe sizes, the worker assignment may be slightly imbalanced.

## CSV / JSON / JSONL: No Sub-File Splitting

Text formats have no footer metadata and no seek-friendly internal structure. Multiple files are still load-balanced across workers by file size using LPT scheduling — the limitation is only that a **single large file cannot be split across workers**.

If you have a few very large CSV or JSONL files, split them into smaller files (128–512 MiB each) to give the strategy enough chunks to distribute evenly.

## Iceberg: Delete Files Require Reading Both Data File and Delete File

When position delete files are present, **two files must be read per data file**:

```
data_file.parquet      ← all rows including "deleted" ones
delete_file.parquet    ← list of (file_path, row_position) pairs to exclude
```

`IcebergDataset` routes through `pyiceberg.io.pyarrow.ArrowScan`, which reads both files, builds an in-memory row-position mask, and filters before yielding batches. This is unavoidable — pyarrow has no knowledge of Iceberg delete files and would return deleted rows if used directly.

**Performance implications:**
- Each worker reconnects to the Iceberg catalog per file to reconstruct the `FileScanTask` (catalog round-trip per file, not once at startup).
- I/O is higher than clean tables: every data file read is accompanied by a delete file read.
- For large tables with many small delete files, this can dominate worker time.

**Workaround:** compact the table before training to merge deletes back into clean data files:

```sql
-- Spark / Trino / Flink
ALTER TABLE my_table EXECUTE optimize;
-- or
CALL system.rewrite_data_files('my_db.my_table');
```

After compaction, `_has_deletes` becomes `False` and the fast path (direct pyarrow reader, sub-file splitting active) is restored.

## Iceberg: Equality Deletes Not Supported

Tables with **equality delete files** (produced by some engines' `DELETE` or `MERGE INTO` operations) will raise `NotImplementedError` from `pyiceberg` — equality deletes are not implemented in the pyiceberg Arrow reader.

**Workaround:** compact the table first (same `rewrite_data_files` command above) to convert equality deletes to position deletes, or rewrite to clean files.

## Iceberg: Sub-File Splitting Disabled with Delete Files

When position delete files are present, `IcebergDataset` falls back to **file-level splits only** — no row group splitting within a file.

**Why:** position delete offsets are absolute row positions within the original data file. If a file is split into sub-ranges (e.g. rows `[0, 1000)` and `[1000, 2000)`), a delete at position 1500 only appears in the second chunk. The worker reading `[0, 1000)` never sees it and silently returns a row that should be deleted.

**Impact:** large data files with delete files cannot be distributed across multiple workers — the entire file is assigned to one worker. For clean tables (no deletes), `TargetSizeSplitStrategy` sub-splits at row group boundaries and each row group can go to a different worker.

**Workaround:** compact the table to remove delete files (see above), or partition large files into smaller ones so the file-level scheduler still distributes work evenly.

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
