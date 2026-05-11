# Spec: ORC Sub-File Splitting

## Core Principle

ORC files have **stripes** — self-contained, independently-readable units analogous to
Parquet row groups. Stripes are the natural split boundary for ORC: no partial stripe reads
are needed, and each stripe can be requested independently via PyArrow's
`ORCFile.read(stripes=[...])`.

Without sub-file splitting, a single large ORC file is treated as one unsplittable chunk
and assigned to one worker. Sub-file splitting lets multiple workers share a large ORC file
by giving each worker a disjoint stripe range.

---

## ORC File Anatomy

```
ORC file layout
├── Stripe 0  [index | columns | stripe footer]
├── Stripe 1  [index | columns | stripe footer]
├── ...
├── Stripe N-1
├── File Footer   ← stripe metadata: nrows per stripe, offsets, byte lengths
└── Postscript    ← compression, footer length
```

The **file footer** contains `StripeInformation` for every stripe, including:
- `numberOfRows` — exact row count for that stripe
- `dataLength` — compressed column data bytes
- `indexLength` — row index bytes

PyArrow reads the file footer on open (`ORCFile(path)`). The public Python API exposes
`orf.nstripes` and `orf.nrows` (total), but not per-stripe row counts as a sequence.

### Metadata API Limitation

Unlike `pq.read_metadata()` (Parquet), PyArrow's `ORCFile` does not expose per-stripe
row counts without reading stripe data. This affects split generation accuracy:

| Data | Parquet | ORC |
|------|---------|-----|
| Total row count | ✓ footer | ✓ footer |
| Per-unit row count | ✓ `rg.num_rows` | ✗ not exposed |
| Per-unit byte size | ✓ `rg.total_byte_size` | ✗ not exposed |

**Practical workaround**: assume uniform stripe sizes — both row count and byte size are
approximated as `total / nstripes`. This is valid for the common case; ORC writers
(Hive, Spark, PyArrow) produce stripes of roughly equal size. When stripes are
non-uniform the approximation may cause minor load imbalance but will never drop rows.

> **Future improvement**: if a future PyArrow release exposes per-stripe row counts via
> `ORCFile`, `_orc_chunks()` can be upgraded to use exact counts without any API change.

---

## Design

### Split Generation — `_orc_chunks()`

New function in `target_size.py`, mirroring `_parquet_chunks()`:

```python
def _orc_chunks(
    file: DataFileInfo,
    target_bytes: int | None,
    target_rows: int | None = None,
) -> Iterator[Split]:
```

**Algorithm**:
1. Open `pyarrow.orc.ORCFile(file.path)` — reads file footer only (no data scan).
2. Compute `rows_per_stripe ≈ nrows / nstripes` and
   `bytes_per_stripe ≈ file.file_size / nstripes`.
3. Pack consecutive stripes into chunks until `chunk_bytes >= target_bytes`
   (or `chunk_rows >= target_rows` when `target_rows` is set).
   Always include at least one stripe per chunk.
4. Each chunk → `Split(file=file, row_range=RowRange(offset=start_row, length=row_count))`.
   `start_row` and `row_count` are derived from the uniform approximation:
   `start_row = first_stripe_index * rows_per_stripe`,
   `row_count = num_stripes_in_chunk * rows_per_stripe`.
   The last chunk gets `row_count = nrows - start_row` to absorb rounding.
5. If `nstripes == 0` or metadata read fails → yield one whole-file `Split` (no `row_range`).

The `RowRange` stores row offsets, not stripe indices, for consistency with Parquet. The
reader maps back to stripe indices using the same uniform formula.

### Reading — `_read_orc_row_range()`

New function in `reader.py`, mirroring `_read_parquet_row_range()`:

```python
def _read_orc_row_range(
    original_path: str,
    resolved_path: str,
    row_range: RowRange,
    columns: list[str] | None,
    filters: pc.Expression | None,
    batch_size: int,
    arrow_fs: pafs.FileSystem | None,
    partitioning: str | None,
) -> Iterator[pa.RecordBatch]:
```

**Algorithm**:
1. Open `ORCFile(resolved_path)` — reads file footer.
2. Compute `rows_per_stripe = orf.nrows / orf.nstripes` (same formula as generation).
3. Derive stripe range:
   `start_stripe = row_range.offset // rows_per_stripe`
   `end_stripe = ceil((row_range.offset + row_range.length) / rows_per_stripe)`
   Clamp to `[0, nstripes)`.
4. Read: `table = orf.read(stripes=list(range(start_stripe, end_stripe)), columns=columns)`.
5. Apply `filters` post-read via `table.filter(filters)`.
6. Attach Hive partition columns (same as `_read_parquet_row_range`).
7. Yield in `batch_size`-row slices.

**No data is read for stripes outside the assigned range** — `orf.read(stripes=[...])` is
true random access.

### Dispatch in `read_split()`

Extend the existing dispatch in `reader.py` line 107:

```python
# Before (parquet only):
if split.row_range is not None and arrow_format == "parquet":

# After (parquet + orc):
if split.row_range is not None and arrow_format in ("parquet", "orc"):
```

Route ORC `row_range` splits to `_read_orc_row_range()`.

### Strategy Dispatch in `target_size.py`

Add ORC branch alongside Parquet in `TargetSizeSplitStrategy.generate()`:

```python
for file in files:
    ext = file.path.rsplit(".", 1)[-1].lower() if "." in file.path else ""
    if ext == "parquet":
        all_splits.extend(_parquet_chunks(file, self.target_bytes, self.target_rows))
    elif ext == "orc":                                          # new
        all_splits.extend(_orc_chunks(file, self.target_bytes, self.target_rows))
    else:
        all_splits.append(Split(file=file, row_range=None))
```

---

## Requirements

### Sub-File Splitting `[v2]`
The system SHALL split ORC files at stripe boundaries.
The system SHALL NOT split a stripe across two chunks.
The system SHALL produce at least one chunk per non-empty ORC file.

### Metadata Read `[v2]`
Split generation SHALL read only the ORC file footer (no stripe data scan).
If the footer is unreadable, the system SHALL fall back to a single whole-file chunk.

### Row Range Consistency `[v2]`
`RowRange.offset` and `RowRange.length` stored at split time SHALL use the same
`nrows / nstripes` approximation used at read time so stripe indices are recovered correctly.
The last chunk of each file SHALL absorb the rounding remainder so no rows are dropped.

### Read Correctness `[v2]`
`_read_orc_row_range()` SHALL read exactly the stripes that correspond to the assigned
`RowRange` — no stripes outside the range, no omitted stripes within it.
Column projection and predicate filter SHALL be applied (post-read filter for ORC).

### Fallback `[v2]`
ORC files with `nstripes == 0` SHALL be yielded as a single whole-file chunk (no `row_range`).
ORC files that fail metadata read SHALL be yielded as a single whole-file chunk.

---

## Files to Change

| File | Change |
|------|--------|
| `src/torch_dataloader_utils/splits/target_size.py` | Add `_orc_chunks()`, add `elif ext == "orc"` in `generate()` |
| `src/torch_dataloader_utils/format/reader.py` | Add `_read_orc_row_range()`, extend dispatch to `"orc"` |
| `tests/unit/splits/test_target_size.py` | Add ORC chunking tests (mocked `ORCFile`) |
| `tests/unit/format/test_reader.py` | Add `_read_orc_row_range` unit tests |
| `tests/integration/test_local.py` | Add ORC integration scenario |

---

## Scenarios

**Split generation** — single ORC file, 8 stripes, ≈50 MiB each, `target_bytes=128 MiB`, `num_workers=2`

| Scenario | Input | Expected chunks |
|----------|-------|-----------------|
| Target 128 MiB, 8 × 50 MiB stripes | 1 file | 4 chunks (2–3 stripes each) |
| `target_rows=10000`, 8 stripes × 2500 rows | 1 file | 4 chunks (2 stripes = 5000 rows each) |
| Single stripe file | 1 file | 1 chunk (whole file, 1 stripe) |
| `nstripes=0` (empty ORC) | 1 file | 1 whole-file chunk (fallback) |
| Metadata read fails | 1 file | 1 whole-file chunk (fallback) |
| 2 ORC files, 4 stripes each, `num_workers=2` | 2 files | 8 chunks → 4 per worker |

**Correctness invariants**
- Union of all chunk row ranges = full file row range, no gaps, no overlaps
- Chunk count for single file ≈ `ceil(file_size / target_bytes)`
- Last chunk absorbs rounding — `sum(chunk.row_range.length) == nrows`

**Multi-file** — mixed Parquet + ORC, `num_workers=2`

| File mix | Expected behaviour |
|----------|--------------------|
| 1 Parquet + 1 ORC | Each file chunked independently; all chunks pooled into flat list before LPT assignment |
| 1 ORC + 1 CSV | ORC sub-splits; CSV whole-file; mixed chunks assigned by LPT |

**Read correctness** — `_read_orc_row_range()`

| Scenario | Input | Expected |
|----------|-------|----------|
| First half of file | `RowRange(0, N/2)` | Reads stripes [0, nstripes/2), returns N/2 rows |
| Second half of file | `RowRange(N/2, N/2)` | Reads stripes [nstripes/2, nstripes), returns N/2 rows |
| With column projection | `columns=["a","b"]` | Only columns a, b in output |
| With filter | `filters=pc.field("x") > 0` | Post-read filter applied |
| Hive partitioning | `partitioning="hive"` | Partition columns appended from path |

**Fallback to whole-file**

| Condition | Behaviour |
|-----------|-----------|
| `nstripes == 0` | Single `Split(row_range=None)` — existing whole-file Scanner path |
| `ORCFile(path)` raises | Log warning, single `Split(row_range=None)` |
| `row_range is None` (non-ORC fallback) | Existing `pad.dataset` Scanner path unchanged |

**Validation errors**

| Condition | Error |
|-----------|-------|
| Unreadable ORC file at read time | `pyarrow.lib.ArrowIOError` propagates — not suppressed |
