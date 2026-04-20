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

#### Scenario: Parquet directory — full read
- GIVEN a directory with 3 Parquet files, 100 rows each (300 total)
- WHEN `create_dataloader(path=dir, format="parquet", num_workers=0, batch_size=50)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 300
- AND each batch is a `dict[str, torch.Tensor]`

#### Scenario: ORC directory — full read
- GIVEN a directory with 2 ORC files, 100 rows each (200 total)
- WHEN `create_dataloader(path=dir, format="orc", num_workers=0)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 200

#### Scenario: CSV directory — full read
- GIVEN a directory with 2 CSV files, 100 rows each (200 total)
- WHEN `create_dataloader(path=dir, format="csv", num_workers=0)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 200

#### Scenario: JSONL directory — full read
- GIVEN a directory with 2 JSONL files, 100 rows each (200 total)
- WHEN `create_dataloader(path=dir, format="jsonl", num_workers=0)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 200

#### Scenario: Column projection end-to-end
- GIVEN a Parquet directory with columns [feature_a, feature_b, label]
- WHEN `create_dataloader(..., columns=["feature_a", "label"])` is called
- THEN each batch contains only keys "feature_a" and "label"

#### Scenario: Predicate pushdown end-to-end
- GIVEN a Parquet directory where feature_b values range from 0 to 99 across 100 rows
- WHEN `create_dataloader(..., filters=pc.field("feature_b") >= 50)` is called
- THEN total rows collected equals 50 (only rows where feature_b >= 50)

#### Scenario: Glob pattern
- GIVEN a directory with mixed file types (.parquet and .csv)
- WHEN `create_dataloader(path="dir/*.parquet", format="parquet", num_workers=0)` is called
- THEN only Parquet files are read

#### Scenario: Single file
- GIVEN a single Parquet file path
- WHEN `create_dataloader(path=file, format="parquet", num_workers=0)` is called
- THEN rows from that file are returned

#### Scenario: output_format="numpy"
- GIVEN a Parquet directory
- WHEN `create_dataloader(..., output_format="numpy")` is called
- THEN each batch value is a `np.ndarray`

#### Scenario: Shuffle reproducibility
- GIVEN a Parquet directory with 3 files and `shuffle=True, shuffle_seed=42`
- WHEN two DataLoaders are created with identical parameters
- AND both are fully iterated
- THEN both produce identical row sequences

#### Scenario: No rows dropped or duplicated
- GIVEN a Parquet directory with 3 files, 100 rows each (300 total)
- WHEN `create_dataloader` is iterated
- THEN the union of all row ids across batches equals exactly {0..299} with no duplicates

#### Scenario: Multi-worker — no rows dropped or duplicated
- GIVEN a directory with 4 Parquet files, 100 rows each (400 total)
- WHEN `create_dataloader(..., num_workers=2)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 400
- AND the set of all row_ids equals exactly {0..399} with no duplicates

#### Scenario: Multi-worker — more files than workers
- GIVEN a directory with 6 Parquet files, 100 rows each (600 total)
- WHEN `create_dataloader(..., num_workers=2)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 600 (each worker reads 3 files)
- AND no rows are dropped or duplicated

#### Scenario: Imbalanced files — size-balanced splits
- GIVEN 4 Parquet files with row counts [400, 300, 200, 100] (1000 total)
- AND `num_workers=2`
- WHEN `SizeBalancedSplitStrategy` is auto-selected (files have file_size)
- AND the DataLoader is fully iterated
- THEN total rows collected equals 1000
- AND no rows are dropped or duplicated
- AND the larger worker's row count is no more than 2x the smaller worker's row count
