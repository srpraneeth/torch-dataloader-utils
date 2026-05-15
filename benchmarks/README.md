# Benchmarks

Reproducible benchmark suite for `torch-dataloader-utils`. Measures throughput,
load balance, and I/O efficiency across 8 scenarios.

## Quick start

```bash
# 1. Install benchmark dependencies (psutil for I/O measurement)
pip install -r benchmarks/requirements.txt

# 2. Generate data
python -m benchmarks.gen_data --out-dir /tmp/bench_data --dataset all

# 3. Run all scenarios
python -m benchmarks.run --data-dir /tmp/bench_data

# 4. View results
python -m benchmarks.report benchmarks/results/<timestamp>.json
```

> All commands are run from the repo root with `python -m` — not as scripts.

---

## Step-by-step

### 1. Generate data

```bash
# Single dataset (small = 10 files × 50K rows, ~13MB)
python -m benchmarks.gen_data --out-dir /tmp/bench_data --dataset small

# All dataset sizes (tiny, small, medium, large, unequal, single_large)
python -m benchmarks.gen_data --out-dir /tmp/bench_data --dataset all

# Include ORC and CSV variants (needed for S8 format comparison)
python -m benchmarks.gen_data --out-dir /tmp/bench_data --dataset small --format all
```

Available dataset profiles:

| Name           | Files | Rows/file | Total rows | Use for                                          |
|----------------|-------|-----------|------------|--------------------------------------------------|
| `tiny`         | 4     | 10K       | 40K        | CI — fast, always runs                           |
| `small`        | 10    | 50K       | 500K       | Local dev — quick sanity check                   |
| `large`        | 50    | 500K      | 25M        | Publishing results                               |
| `large_sorted` | 50    | 500K      | 25M        | Predicate pushdown (S6) — rows sorted by label   |
| `unequal`      | 20    | 10K–160K  | 750K       | Load balance scenario (S2)                       |
| `single_large` | 1     | 10M       | 10M        | Sub-file splitting scenario (S3)                 |

Each directory gets a `manifest.json` with row counts and SHA-256 checksums.

### 2. Run benchmarks

```bash
# Run all 8 scenarios, default dataset sizes
python -m benchmarks.run --data-dir /tmp/bench_data

# CI mode: tiny dataset, 3 runs, no warmup (fast)
python -m benchmarks.run --data-dir /tmp/bench_data --ci

# Specific scenarios
python -m benchmarks.run --data-dir /tmp/bench_data --scenarios S1 S2 S3

# More runs for stable numbers
python -m benchmarks.run --data-dir /tmp/bench_data --n-runs 10

# Skip manifest checksum verification
python -m benchmarks.run --data-dir /tmp/bench_data --skip-verify

# Save results to a custom directory
python -m benchmarks.run --data-dir /tmp/bench_data --output-dir benchmarks/results
```

Results are written to `<output-dir>/YYYY-MM-DDTHH-MM-SS.json`.

### 3. View results

```bash
# ASCII table for a single run
python -m benchmarks.report benchmarks/results/2026-05-13T10-00-00.json
```

Example output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  S1  Baseline throughput sweep across num_workers on equal-sized files.
  dataset=small · rows=500K · files=10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  implementation      nw=0        nw=2        nw=4
  this_library        4.0M r/s    2.1M r/s    3.8M r/s
  manual_sharded      1.6M r/s    1.7M r/s    3.0M r/s
  naive_iterable      1.6M r/s    880K r/s    470K r/s
```

### 4. Baseline and regression detection

```bash
# Set a baseline (promote current results)
python -m benchmarks.run --data-dir /tmp/bench_data --update-baseline

# Future runs automatically check against baseline (90% threshold)
python -m benchmarks.run --data-dir /tmp/bench_data
# => "⚠ REGRESSION DETECTED" or "✓ No regressions vs baseline."
```

The baseline is stored at `benchmarks/results/baseline.json` and committed to git.

### 5. Update the docs page

After a run you're happy with, regenerate `docs/benchmarks.md` in one command:

```bash
python -m benchmarks.report benchmarks/results/<timestamp>.json \
  --docs --out docs/benchmarks.md
```

Preview locally:

```bash
.venv/bin/mkdocs serve   # open http://127.0.0.1:8000/benchmarks/
```

---

## Baselines explained

S1 and S3 compare the library against two hand-written implementations:

**`manual_sharded`** — what a careful engineer writes *without* this library. Files are
pre-partitioned across workers at startup: worker 0 gets files 0, N, 2N..., worker 1
gets files 1, N+1, 2N+1..., etc. No I/O waste — each file is read exactly once.
This is the realistic comparison point. It matches the library on equal-sized files,
but falls behind when files are unequal (S2) or when there is a single large file (S3),
because it cannot split a file across multiple workers.

**`naive_iterable`** — the common anti-pattern. Every worker reads every file and
discards batches that aren't assigned to it (`if batch_index % num_workers == worker_id`).
With 4 workers, each reads all 10 files → 4× I/O amplification. Throughput degrades
as you add workers rather than improving.

---

## Scenarios

| ID | Name                  | Dataset               | What it measures                                        |
|----|-----------------------|-----------------------|---------------------------------------------------------|
| S1 | Baseline throughput   | large                 | rows/sec sweep over num_workers; 3 implementations      |
| S2 | Unequal files         | unequal               | per-worker row distribution; std_dev as balance metric  |
| S3 | Single large file     | single_large          | sub-file splitting vs whole-file assignment             |
| S4 | Rank-aware sharding   | large                 | this_library 1× I/O per rank vs naive_ddp N× amplification |
| S5 | Column projection     | large                 | full-schema vs 2-column; bytes-read reduction           |
| S6 | Predicate pushdown    | large + large_sorted  | label==0 filter; row-group pruning on sorted vs uniform data |
| S7 | Startup latency       | tiny/small/large      | create_dataloader() and time-to-first-batch             |
| S8 | Format comparison     | small                 | parquet vs orc vs csv throughput and on-disk size       |

---

## Results file format

```json
{
  "meta": {
    "run_at": "2026-05-13T10:00:00",
    "git_sha": "abc1234",
    "library_version": "0.1.0",
    "platform": "darwin-arm64",
    "python": "3.12.5"
  },
  "scenarios": {
    "S1": {
      "description": "Baseline throughput sweep...",
      "dataset": "small",
      "total_rows": 500000,
      "this_library": {
        "num_workers=4": {
          "elapsed_sec": {"median": 0.13, "p25": 0.12, "p75": 0.14, "min": 0.12},
          "rows_per_sec": {"median": 3800000, "p25": 3600000, "p75": 4200000},
          "total_rows": 500000
        }
      }
    }
  }
}
```

---

## Directory layout

```
benchmarks/
  README.md             — this file
  gen_data.py           — generate synthetic datasets
  run.py                — CLI runner (all scenarios)
  report.py             — ASCII table renderer
  _common.py            — shared utilities (measure, passthrough, ...)
  requirements.txt      — psutil
  results/
    baseline.json       — committed reference; regression gate target
    *.json              — timestamped run outputs (gitignored)
  baselines/
    naive_iterable.py   — anti-pattern: every worker reads every file
    manual_sharded.py   — careful: pre-partitioned file-level sharding
  scenarios/
    s1_throughput.py    — S1
    s2_unequal.py       — S2
    ...
```
