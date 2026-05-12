# Spec: Split Generation

## Core Principle

Split generation has three distinct concerns that must not be mixed:

```
File Discovery  →  Predicate Pushdown  →  Split Balancing  →  Split Assignment
(filesystem)       (pyarrow/pyiceberg)    (strategy)          (LPT heap / rank slice)
```

Predicate pushdown happens **before** splits — it reduces the file list. The split strategy only ever sees files that survived filtering.

---

## Data Classes

### `DataFileInfo` `[v1]`
Carries metadata for plain files (Parquet, ORC, CSV, JSON).

```
path: str                      # file URI
file_size: int | None          # bytes — from fsspec stat()
record_count: int | None       # rows — not available for plain files in V1
```

### `IcebergDataFileInfo(DataFileInfo)` `[v1]`
Extends `DataFileInfo` with Iceberg manifest metadata.

```
partition: dict[str, str] | None    # e.g. region=US, date=2024-01-01
snapshot_id: int | None             # for reproducibility and time travel
```

Note: `column_stats` are NOT stored here — column-level predicate pushdown is handled by `pyiceberg` during file discovery, before files reach the split layer.

### `RowRange` `[v1]`
Defines a row-level slice within a single file.

```
offset: int    # start row (inclusive)
length: int    # number of rows to read
```

### `FileSplit` `[v1]`
Pairs a file with an optional row range.

```
file: DataFileInfo
row_range: RowRange | None    # None = read entire file
                              # RowRange = sub-file slice (Parquet row groups or ORC stripes)
```

### `Split` `[v1]`
A unit of work assigned to one DataLoader worker.

```
id: int
file_splits: list[FileSplit]    # files (and optional row ranges) this worker reads
row_count: int | None           # total rows across all file_splits (optional)
size_bytes: int | None          # total bytes across all file_splits (optional)
```

---

## Requirements

### `RoundRobinSplitStrategy` `[v1]`
The system SHALL distribute files across splits using round-robin assignment.
The system SHALL use this strategy when the file list is empty (fallback only).
The system SHALL ignore `file_size` and `record_count` metadata.
The system SHALL produce exactly N splits where N equals `num_workers`.
No file SHALL appear in more than one split.

### `TargetSizeSplitStrategy` `[v1 — default strategy]`
The system SHALL be the default auto-selected strategy for non-empty file lists.

**Parquet chunking:**
The system SHALL read row group metadata from the Parquet file footer once in the main process (no data scan).
The system SHALL pack consecutive row groups into chunks targeting `target_bytes` (default 128 MiB).
When `target_rows` is set it SHALL take precedence over `target_bytes`.
Each chunk SHALL be a `FileSplit` with a `RowRange(offset, length)` in rows.
No row group SHALL be split across two chunks.

**ORC chunking `[v2]`:**
The system SHALL split ORC files at stripe boundaries.
Because PyArrow does not expose per-stripe row counts, the row count SHALL be approximated uniformly as `total_rows / num_stripes`.
Each chunk SHALL be a `FileSplit` with a `RowRange` covering the assigned stripe indices.

**CSV / JSON / JSONL:**
Each file SHALL be treated as a single unsplittable chunk — `row_range=None`.

**Assignment:**
All chunks SHALL be collected into a flat list (potentially many more than `num_workers`).
Chunks SHALL be assigned to workers using a greedy min-heap (LPT scheduling) — always assign the next chunk to the least-loaded worker.
This is optimal for unequal chunk sizes and equivalent to round-robin for equal sizes.

### Strategy Auto-Selection `[v1]`
The system SHALL auto-select the split strategy when `split_strategy=None`:
- Non-empty file list → `TargetSizeSplitStrategy`
- Empty file list → `RoundRobinSplitStrategy`
The system SHALL always allow the user to override via `split_strategy=`.
The system SHALL log the selected strategy at `INFO` level.

### `SplitStrategy` Protocol `[v1]`
The system SHALL define `SplitStrategy` as a `Protocol` — not an ABC.
Any class with a matching `generate()` method SHALL satisfy the protocol.
No inheritance SHALL be required from user-defined strategies.

V2 signature (with rank-aware sharding):
```python
class SplitStrategy(Protocol):
    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int,
        num_ranks: int = 1,
        rank: int = 0,
    ) -> list[Shard]:
        ...
```

V1 strategies that do not accept `num_ranks` / `rank` SHALL still work — the system detects the missing parameters via `inspect.signature` and omits them from the call.

### Rank-Aware Sharding `[v2]`
Both `TargetSizeSplitStrategy` and `RoundRobinSplitStrategy` SHALL accept `num_ranks: int = 1` and `rank: int = 0` parameters on `generate()`.

After all chunks are generated (and optionally shuffled), the system SHALL slice them for the current rank using interleaved assignment:
```
rank_splits = all_splits[rank::num_ranks]
```

This gives rank 0 splits at indices 0, num_ranks, 2×num_ranks, …; rank 1 gets indices 1, num_ranks+1, …; etc. Interleaved assignment distributes large and small chunks evenly across ranks without sorting.

The system SHALL raise `ValueError` when `not (0 <= rank < num_ranks)`.
The default `num_ranks=1, rank=0` SHALL produce identical behaviour to V1 — all splits go to the single rank.

### Shuffle `[v1]`
The system SHALL shuffle the chunk list before rank slicing and worker assignment when `shuffle=True`.
The system SHALL use `shuffle_seed + epoch` as the random seed.
The system SHALL NOT mutate the input file list.
`set_epoch()` SHALL always regenerate splits regardless of `shuffle` setting — with `shuffle=False` the output is deterministic and identical each call.

### `num_workers` Auto-Detection `[v1]`
The system SHALL accept `num_workers=None` to trigger auto-detection.
Auto-detection SHALL use `max(1, os.cpu_count() - 1)`.
The system SHALL log the resolved value at `INFO` level.
The system SHALL treat `num_workers=0` as single-process mode (PyTorch convention).

### Split Timing `[v1]`
File discovery SHALL happen once at `create_dataloader()` time.
Split generation SHALL happen at construction time and on every `set_epoch()` call.

---

## Scenarios

**TargetSizeSplitStrategy — Parquet row group packing**

| Files | Workers | Expected |
|-------|---------|----------|
| 1 file, 10 row groups of 10 MiB each, `target_bytes=128MiB` | 1 | 1 chunk covering all 10 row groups |
| 1 file, 10 row groups of 20 MiB each, `target_bytes=64MiB` | 4 | Multiple chunks, each ≤ 2 row groups |
| 3 files (100, 1 000, 10 000 rows), `num_workers=2` | 2 | Both workers get ≈ equal row counts via LPT |

**ORC stripe splitting**

| Files | Workers | Expected |
|-------|---------|----------|
| 1 ORC file, 4 stripes, `num_workers=4` | 4 | Each worker gets 1 stripe; no rows dropped or duplicated |
| 1 ORC file, 3 stripes, `num_workers=2` | 2 | One worker gets 2 stripes, one gets 1 |

**RoundRobin distribution**

| Files | Workers | Expected |
|-------|---------|----------|
| 8 equal files | 4 | Each split gets exactly 2 files |
| 9 files | 4 | No file in more than one split; largest split has at most 1 more file than smallest |

**Strategy auto-selection**

| File metadata | Selected strategy |
|---------------|------------------|
| Non-empty Parquet or ORC file list | `TargetSizeSplitStrategy` |
| Empty file list | `RoundRobinSplitStrategy` |

**Rank-aware sharding**

| Chunks | num_ranks | Expected |
|--------|-----------|----------|
| `[C0,C1,C2,C3,C4,C5]` | 3 | Rank 0→`[C0,C3]`, Rank 1→`[C1,C4]`, Rank 2→`[C2,C5]` |
| 2 chunks, 4 ranks | 4 | Rank 0→`[C0]`, Rank 1→`[C1]`, Rank 2→`[]`, Rank 3→`[]` |
| `rank >= num_ranks` | any | `ValueError` |
| `rank < 0` | any | `ValueError` |

**Shuffle** — `shuffle=True, seed=42`: same epoch → identical split assignments; epoch 0 vs epoch 1 → different chunk order

**No shuffle** — `shuffle=False`: `set_epoch()` called twice → same deterministic split order both times

**Custom strategy** — user-defined class with `generate()` method accepting only `(files, num_workers, epoch)` → V1 signature detected via `inspect.signature`; `num_ranks`/`rank` not passed; strategy receives all chunks

**Sub-file splitting** — 1 Parquet file, `record_count=1_000_000`, 4 workers → 4 `FileSplit`s each with `RowRange`, no rows overlap or gap
