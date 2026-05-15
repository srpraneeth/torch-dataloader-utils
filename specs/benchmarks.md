# Spec: Reproducible Benchmarks

## Core Principle

Benchmarks answer one question: **how much faster is this library than the alternatives, and does it stay that way?** Every benchmark must be runnable by anyone with the repo — no cloud credentials, no GPU, no special hardware. Results are deterministic across runs on the same machine and tracked over time to catch regressions.

Three things are measured:

1. **Throughput** — rows/sec delivered to the training loop
2. **I/O amplification** — bytes read from storage per sample (1.0× = perfect, each byte read once)
3. **Startup latency** — seconds from `create_dataloader()` call to first batch received

---

## Directory Layout

```
benchmarks/
  gen_data.py          # generate all benchmark datasets (extends e2e/gen_data.py schema)
  run.py               # CLI entry point — select scenario, baseline, output format
  scenarios/
    throughput.py      # rows/sec and batches/sec measurement harness
    io_amp.py          # bytes-read instrumentation harness
    startup.py         # time-to-first-batch harness
    load_balance.py    # per-worker row count variance harness
  baselines/
    naive_iterable.py  # every worker reads every file, filters by index parity
    manual_sharded.py  # careful manual file partitioning (no sub-file splitting)
    huggingface_ds.py  # HuggingFace datasets baseline (optional, gated on install)
  results/
    baseline.json                  # committed reference results (updated intentionally)
    2026-05-12T14-32-00.json       # each run writes a new timestamped file
    2026-05-13T09-15-42.json       # old runs are kept — delete manually when needed
  report.py            # render results as Markdown / ASCII table / CSV
  requirements.txt     # pinned versions for all benchmark dependencies
  README.md            # how to run, how to interpret, how to update baselines
```

---

## Synthetic Dataset

All benchmarks use synthetic data generated from a fixed seed. The schema is fixed — changing it would invalidate historical comparisons.

### Schema

```
row_id:      int32     — unique row identifier (used to verify no drops/duplicates)
feat_00..63: float32   — 64 float features (realistic embedding input size)
label:       int32     — 0–9 class label
```

Same schema as `e2e/gen_data.py`. Row values are deterministic: `feat_i = float((row_id + i) % 100) / 100.0`.

### Dataset Sizes

| Name | Files | Rows/file | Row group size | Total rows | Approx size |
|------|-------|-----------|---------------|------------|-------------|
| `tiny` | 4 | 10 000 | 2 000 | 40 000 | ~10 MiB |
| `small` | 10 | 50 000 | 5 000 | 500 000 | ~130 MiB |
| `medium` | 20 | 200 000 | 20 000 | 4 000 000 | ~1 GiB |
| `large` | 50 | 500 000 | 50 000 | 25 000 000 | ~6.5 GiB |
| `unequal` | 20 | 10 000–80 000 (geometric) | 10 000 | ~600 000 | ~160 MiB |
| `single_large` | 1 | 2 000 000 | 20 000 | 2 000 000 | ~530 MiB |

**CI uses `tiny`.** All others are for local / dedicated benchmark runs.

### Data Generation

```bash
python benchmarks/gen_data.py --out-dir /tmp/bench_data --dataset small
```

The generator writes a `manifest.json` alongside the files containing:
- file names, sizes, row counts
- sha256 of each file
- generation timestamp and seed

On each benchmark run, the manifest is verified before measuring. If checksums do not match, the run aborts with an error — this prevents silently benchmarking different data than what was used for the baseline.

---

## Baselines

All baselines receive the exact same data directory, `num_workers`, `batch_size`, and `columns`. The only variable is the data loading implementation.

### `this_library` (the subject)

```python
loader, _ = StructuredDataset.create_dataloader(
    path=data_dir, format="parquet",
    num_workers=num_workers, batch_size=batch_size, columns=columns,
)
```

### `naive_iterable`

Every worker reads every file and skips rows that don't belong to it. Represents the naïve baseline that motivated this library.

```python
class NaiveDataset(IterableDataset):
    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for path in sorted(glob(f"{data_dir}/*.parquet")):
            for i, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size)):
                if i % nw == wid:
                    yield batch
```

This is a reasonable approximation of the "filter by batch index" pattern. I/O amplification should be ~`num_workers`×.

### `manual_sharded`

Files are pre-partitioned across workers at DataLoader startup using `get_worker_info()`. Correct per-file sharding with no sub-file splitting. Represents what a careful engineer writes without this library — expected to match our throughput for equal-sized files but diverge for unequal files.

```python
class ManualShardedDataset(IterableDataset):
    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        my_files = sorted(glob(f"{data_dir}/*.parquet"))[wid::nw]
        for path in my_files:
            yield from pq.ParquetFile(path).iter_batches(batch_size)
```

I/O amplification should be 1.0× (no waste), but load balance degrades for unequal files.

---

## I/O Instrumentation

Measuring bytes actually read from storage is critical for the amplification metric. We do not use `strace` or OS-level tools (not portable). Instead we wrap the fsspec filesystem with a byte counter injected at benchmark time.

```python
class CountingFS:
    """Wraps any fsspec AbstractFileSystem and counts bytes read."""
    def __init__(self, inner):
        self._inner = inner
        self.bytes_read = 0

    def open(self, path, mode="rb", **kwargs):
        f = self._inner.open(path, mode, **kwargs)
        return _CountingFile(f, self)
    
    def __getattr__(self, name):
        return getattr(self._inner, name)
```

`CountingFS` is injected via `storage_options={"_counting_fs": counter}` for the library under test, and wraps the `pyarrow.parquet.ParquetFile` constructor directly for the baselines (via monkeypatching in the harness). The baseline byte-count path must be equivalent — this is a key reproducibility requirement.

The amplification ratio is:

```
amplification = total_bytes_read / sum(file_sizes)
```

Ideal = 1.0. The `naive_iterable` baseline should show ≈ `num_workers`. Column projection and predicate pushdown scenarios should show < 1.0 (fewer bytes than full files).

---

## Measurement Protocol

### Warm-up

Before timing, run one full pass through the dataset to:
- populate OS page cache
- trigger any JIT compilation (torch, pyarrow)
- establish fsspec connection pool

Warm-up results are **not recorded**.

### Timing

Each scenario runs `N_RUNS = 5` timed passes. Record wall-clock time for each pass using `time.perf_counter()`. Report:
- **Median** — primary comparison metric
- **IQR** (p25–p75) — spread indicator
- **Min** — best-case (useful for detecting outliers from OS scheduling)

Do not report mean — it is sensitive to single slow runs caused by GC or OS scheduling.

### Isolation requirements

- No other benchmark or test process running concurrently
- `num_workers > 0` benchmarks must measure the full DataLoader iteration (not just dataset construction)
- The timer wraps the full epoch: from the first `for batch in loader:` call to the last batch received
- GPU is **not required** — the training loop is simulated by counting rows: `total_rows += len(batch["row_id"])`
- After each timed pass, assert `total_rows == expected_rows` — verify no drops

### Row count verification

Every benchmark pass verifies that the exact expected number of rows was yielded. This is the correctness gate — if a baseline silently drops rows, its throughput numbers are meaningless.

---

## Scenarios

### S1 — Baseline throughput (primary)

Sweep `num_workers` across [0, 2, 4, 8] with `small` dataset. Run all three implementations.

**Expected**: `this_library` ≥ `manual_sharded` ≥ `naive_iterable` at every worker count.

| Implementation | num_workers | rows/sec | amplification |
|----------------|-------------|----------|---------------|
| this_library | 0, 2, 4, 8 | ? | ? |
| manual_sharded | 0, 2, 4, 8 | ? | ? |
| naive_iterable | 0, 2, 4, 8 | ? | ? |

### S2 — Unequal file sizes (load balance)

Use `unequal` dataset, `num_workers=4`. Compare per-worker row counts and total wall time.

**Expected**: `this_library` finishes earlier than `manual_sharded` because LPT scheduling distributes row groups, not whole files. `naive_iterable` also finishes slowly because the largest file dominates one worker.

### S3 — Single large file (sub-file splitting)

Use `single_large` dataset, `num_workers=4`.

**Expected**: Only `this_library` can parallelise this at all — it splits the file across 4 workers via row group ranges. `manual_sharded` assigns the whole file to worker 0; workers 1–3 idle. `naive_iterable` reads the file 4 times.

### S4 — Rank-aware sharding (multi-rank simulation)

Use `small` dataset, sweep `num_ranks` across [1, 2, 4] with `num_workers=4`, `rank=0`. Uses the library's built-in rank sharding — no actual distributed process group needed.

**Expected**: rows/sec per rank stays constant (each rank reads only its slice), total bytes read scales as 1/num_ranks.

### S5 — Column projection

Use `small` dataset, `num_workers=4`. Compare `columns=None` vs `columns=["row_id", "label"]` (2 of 66 columns).

**Expected**: ~33× reduction in bytes read (2/66 columns), near-proportional throughput increase.

### S6 — Predicate pushdown

Use `small` dataset with `filters=pc.field("label") == 0` (selects ~10% of rows). Measure bytes read vs no filter.

**Expected**: Rows returned = 10% of total. Bytes read reduction depends on row group selectivity — row groups where `label` is never 0 are skipped at the Parquet statistics level.

### S7 — Startup latency

Measure time from `create_dataloader()` to receipt of first batch, across dataset sizes [tiny, small, medium, large].

**Expected**: `this_library` startup latency grows with file count (metadata scan), but is sub-second for `small` and a few seconds for `large`. `manual_sharded` has near-zero startup (no metadata scan). This is an acknowledged tradeoff — document it.

### S8 — Format comparison (Parquet vs ORC vs CSV)

Use equivalent `small` dataset in Parquet, ORC, and CSV formats. Single implementation (`this_library`), `num_workers=4`.

**Expected**: Parquet ≥ ORC > CSV (due to sub-file splitting availability and columnar encoding efficiency). CSV shows flat throughput scaling with workers (whole-file granularity).

---

## Output Format

Each benchmark run writes to `benchmarks/results/<timestamp>.json` (e.g. `2026-05-12T14-32-00.json`). The regression gate compares against `benchmarks/results/baseline.json`:

```json
{
  "run_at": "2026-05-12T14:32:00",
  "git_sha": "abc1234",
  "library_version": "0.2.0",
  "platform": "darwin-arm64",
  "python": "3.12.3",
  "scenarios": {
    "S1_baseline_throughput": {
      "description": "Throughput sweep across num_workers on equal-sized files. Primary comparison between all three implementations. Higher rows/sec is better.",
      "dataset": "small",
      "this_library": {
        "num_workers=4": {
          "rows_per_sec": {"median": 1250000, "p25": 1180000, "p75": 1310000},
          "amplification": 1.01,
          "total_rows": 500000
        }
      },
      "manual_sharded": { ... },
      "naive_iterable": { ... }
    },
    ...
  }
}
```

`report.py` renders this as an ASCII table or Markdown for pasting into release notes.

---

## CI Integration

CI runs only `tiny` dataset to keep the job under 60 seconds. It does **not** assert absolute performance (hardware varies) but asserts:

1. **Correctness** — all scenarios yield the expected row count
2. **Amplification** — `this_library` amplification ≤ 1.05× (allows 5% overhead for metadata reads)
3. **Relative ordering** — `this_library` rows/sec ≥ `manual_sharded` rows/sec × 0.85 (allow 15% slack for scheduling noise on CI hardware)
4. **Regression gate** — if `baseline.json` exists, rows/sec must be ≥ `baseline × 0.90` (10% regression threshold)

When the regression gate fails, CI prints which scenario regressed and by how much. Updating the baseline requires a deliberate command (`python benchmarks/run.py --update-baseline`) that must be run locally and committed. The timestamped result files are gitignored — only `baseline.json` is committed.

---

## Reproducibility Checklist

- [ ] `benchmarks/requirements.txt` pins all benchmark dependencies (pyarrow, torch, psutil, etc.)
- [ ] `gen_data.py` uses a fixed seed (`BENCH_SEED = 42`) and writes `manifest.json` with checksums
- [ ] All benchmark runs verify checksums before measuring
- [ ] Timing uses `time.perf_counter()` (monotonic, high resolution)
- [ ] Results JSON includes git SHA, library version, platform, and Python version
- [ ] Baseline JSON is committed to the repo and updated only intentionally
- [ ] `README.md` documents how to reproduce results and what hardware the committed baseline was measured on

---

## Scenarios NOT in Scope

**Real S3 / GCS benchmarks** — the I/O amplification ratio (1× vs N×) is algorithmic and identical on S3 and NVMe — the storage backend does not change the result. Absolute rows/sec will be lower on S3 but the relative comparison between implementations moves in our favor (better prefetch). Beyond the algorithmic claim, S3 numbers go stale fast: AWS changes throughput characteristics, results vary by region and instance type, and committed numbers mislead users within months. The only thing real S3 would reveal is fsspec connection pooling behavior, which is a one-time investigation, not a recurring benchmark. If cloud benchmarks are ever needed (blog post, release notes), run them separately and do not commit the results.

**GPU utilization / training loop stall** — a simulated stall fraction (DataLoader wait time divided by wait + `time.sleep` compute budget) only means something relative to the compute budget you choose. Pick a slow model and the DataLoader is obviously never the bottleneck; pick a fast one and it obviously is. The result restates the rows/sec number with an extra free parameter baked in. In practice, users already know whether their DataLoader is the bottleneck: GPU utilization above ~85% means it is not. The rows/sec number we already measure is the actionable metric — users multiply by their batch size and compare to their training step rate themselves.

**Record-level shuffle overhead** — not yet implemented; benchmark when the feature ships.
