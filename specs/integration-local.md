# Spec: Local Filesystem Integration Tests

## Core Principle

Integration tests exercise the full pipeline end-to-end — no mocking, no unit test shortcuts.
A real user calling `StructuredDataset.create_dataloader()` with local files should work exactly
as these tests demonstrate.

```
local directory / file
        ↓
StructuredDataset.create_dataloader()
        ↓
DataLoader iteration
        ↓
batches in requested output format
```

---

## Requirements

### Test Data `[v1]`
Tests SHALL use a realistic local dataset — multiple files, multiple formats.
Test data SHALL be generated fresh in a `tmp_path` fixture — not reuse the small unit fixtures.
Each test file SHALL have at least 100 rows to exercise batching realistically.
Tests SHALL cover all supported formats: Parquet, ORC, CSV, JSONL.

### End-to-End Scenarios `[v1]`
Tests SHALL call `StructuredDataset.create_dataloader()` — the public API entry point.
Tests SHALL iterate the DataLoader fully and verify total row count.
Tests SHALL verify correct output types per `output_format`.
Tests SHALL verify column projection returns only requested columns.
Tests SHALL verify predicate pushdown filters rows correctly.
Tests SHALL verify multi-file directories are fully read (all files, all rows).
Tests SHALL verify shuffle produces consistent results with the same seed.

### Multi-File `[v1]`
Tests SHALL use directories with multiple files (at least 3).
Tests SHALL verify total rows across all files equals expected count.
Tests SHALL verify no rows are duplicated or dropped.

### Format Coverage `[v1]`
Tests SHALL have at least one end-to-end test per format:
- Parquet
- ORC
- CSV
- JSONL

---

## Scenarios

**Format coverage** — `num_workers=0`, `batch_size=50`

| Format | Files | Total rows | Expected |
|--------|-------|------------|----------|
| Parquet | 3 × 100 rows | 300 | 300 rows collected, each batch is `dict[str, torch.Tensor]` |
| ORC | 2 × 100 rows | 200 | 200 rows collected |
| CSV | 2 × 100 rows | 200 | 200 rows collected |
| JSONL | 2 × 100 rows | 200 | 200 rows collected |

**Multi-worker distribution**

| Files | `num_workers` | Expected |
|-------|---------------|----------|
| 4 Parquet × 100 rows | 2 | 400 total rows, set of row_ids = {0..399}, no duplicates |
| 6 Parquet × 100 rows | 2 | 600 total rows, each worker reads 3 files |
| 1 Parquet (1 split) | 4 | 1 split assigned to worker 0; workers 1–3 yield nothing |

**Feature behaviors**

**Column projection** — `columns=["feature_a", "label"]` on `[feature_a, feature_b, label]` → each batch contains only those two keys

**Predicate pushdown** — `filters=pc.field("feature_b") >= 50` on values 0–99 (100-row file) → 50 rows returned

**Glob pattern** — `dir/*.parquet` on directory with mixed `.parquet` and `.csv` → only Parquet files read

**Single file** — direct file path → rows from that file only

**`output_format="numpy"`** — numeric columns are `np.ndarray`

**Shuffle reproducibility** — `shuffle=True, seed=42`: two loaders at same epoch → identical row sequences

**No rows dropped or duplicated** — 3 files × 100 rows, row_id 0–299 → set of all row_ids equals exactly {0..299}

**Imbalanced files** — files with row counts `[400, 300, 200, 100]`, `num_workers=2` → 1000 total rows, no duplicates, larger worker's count ≤ 2× smaller worker's count
