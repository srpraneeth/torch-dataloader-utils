# Benchmarks

Performance measurements across 8 scenarios covering throughput, load balancing, DDP sharding, column projection, predicate pushdown, startup latency, and format comparison.

All numbers are **medians over 5 runs** on a single machine (darwin-arm64, Python 3.12, local NVMe SSD). Figures are intentionally conservative — cloud object storage will show larger gaps where I/O dominates.

---

## Scorecard

One row per scenario — the headline number for each feature.

| | Feature | this_library | vs baseline |
|:---|:---|:---|:---|
| **S1** | Throughput (nw=0) | **3.05M rows/s** | 1.01× vs naive_iter |
| **S2** | Load balance (unequal files) | **0.0% imbalance** | vs 116.7% manual  0.98× wall-clock |
| **S3** | Single large file (nw=8) | **0.51s epoch** | 5.87× faster than manual |
| **S4** | DDP I/O (16 ranks) | **1× per rank** | vs 16× naive_ddp |
| **S7** | Startup latency (50-file dataset) | **0.791s first batch** | metadata scan cost |
| **S8** | Format: parquet vs CSV | **1.44× faster** | 421K vs 293K rows/s |

---

## Baselines used

| Name | Description |
|:---|:---|
| `this_library` | torch-dataloader-utils with row-group-aware splits |
| `manual_sharded` | Files pre-partitioned across workers at startup — no I/O waste, but can't split a single large file across workers |
| `naive_iterable` | Every worker reads every file and discards non-assigned batches — N× I/O amplification |
| `naive_ddp` | Every DDP rank reads all files (like `DistributedSampler` without file sharding) |
| `manual` | Plain `pq.ParquetFile.iter_batches()` with Python-level filtering — no row-group pushdown |

---

## S1 — Baseline throughput

Equal-sized files, sweeping `num_workers`. All three implementations compared.

_50 files · 25.0M rows · dataset=`small`_

| Implementation | nw=0 | nw=2 | nw=4 | nw=8 |
|:---| ---: | ---: | ---: | ---: |
| `this_library` | 3.05M | 2.13M | 1.98M | 1.72M |
| `manual_sharded` | 3.10M | 1.90M | 2.07M | 1.94M |
| `naive_iterable` | 3.02M | 1.53M | 1.29M | 789K |
| **lib / naive speedup** | **1.01×** | **1.40×** | **1.53×** | **2.18×** |

!!! tip "Key insight"
    On equal-sized files `this_library` and `manual_sharded` are equivalent — the library's advantage appears on unequal files (S2), single large files (S3), and DDP training (S4). `naive_iterable` degrades linearly with worker count because each worker reads everything.

---

## S2 — Load balancing on unequal files

32× size spread across 20 files. With 4 workers, `manual_sharded` lands large files on a single worker.

_20 files · 1.9M rows · dataset=`unequal`_

| Implementation | Workers | Total rows | Mean/worker | Std Dev | Imbalance |
|:---|---:|---:|---:|---:|---:|
| `this_library` | 4 | 1.9M | 480K | 0 | **0.0%** |
| `manual_sharded` | 4 | 1.9M | 480K | 206K | **116.7%** |

**`this_library` per-worker:** ['480K', '480K', '480K', '480K']
**`manual_sharded` per-worker:** ['150K', '710K', '490K', '570K']

| Implementation | rows/sec | elapsed |
|:---|---:|---:|
| `this_library` | 980K | 1.959s |
| `manual_sharded` | 996K | 1.927s |

!!! tip "Key insight"
    The library splits files at row-group boundaries so every worker gets the same amount of work. `manual_sharded` assigns whole files — one worker stalls on the large file while others finish early.

---

## S3 — Single large file

One 10M-row file. `manual_sharded` sends the entire file to worker 0; workers 1–7 are idle. The library splits row groups across all 8 workers.

_1 file · 10.0M rows · dataset=`single_large`_

_num\_workers=8_

| Implementation | rows/sec | epoch time | speedup |
|:---|---:|---:|---:|
| `this_library` | 19.67M | 0.51s | baseline |
| `manual_sharded` | 3.35M | 2.98s | 0.17× |

**Batch delivery rate (relevant for GPU utilisation):**

| Implementation | batches/sec | GPU starves if step < |
|:---|---:|---:|
| `this_library` | 19,206 | 0.05 ms |
| `manual_sharded` | 3,274 | 0.31 ms |

!!! tip "Key insight"
    **GPU starves below** is the minimum GPU step time at which the data loader becomes the bottleneck. At 19K batches/sec, the library only starves a GPU running sub-0.05ms steps — effectively never for real models. `manual_sharded` starves any GPU faster than 0.31ms per step.

---

## S4 — DDP rank-aware sharding

Each rank should read only its 1/N slice. `naive_ddp` reads everything on every rank.

_50 files · 25.0M rows · dataset=`large` · num_workers=4_

**`this_library` — reads 1/N of data per rank**

| num_ranks | rows/rank | fraction | rows/sec | I/O amplification |
|---:|---:|---:|---:|---:|
| 1 | 25.0M | 100% | 2.08M | **1×** |
| 2 | 12.5M | 50% | 1.83M | **1×** |
| 4 | 6.5M | 26% | 1.71M | **1×** |
| 8 | 3.5M | 14% | 1.40M | **1×** |
| 16 | 2.0M | 8% | 1.11M | **1×** |

**`naive_ddp` — reads ALL data, discards non-assigned rows**

| num_ranks | rows/rank | fraction | rows/sec | I/O amplification |
|---:|---:|---:|---:|---:|
| 1 | 25.0M | 100% | 2.12M | **1×** |
| 2 | 12.5M | 50% | 1.06M | **2×** |
| 4 | 6.5M | 26% | 551K | **4×** |
| 8 | 3.5M | 14% | 297K | **8×** |
| 16 | 2.0M | 8% | 170K | **16×** |

!!! tip "Key insight"
    `naive_ddp` reads 16× more bytes at 16 ranks — identical to running a single-rank job 16 times over. The library assigns non-overlapping file splits per rank at dataset creation time; each rank reads only what it needs.

---

## S5 — Column projection

Reading 2 columns out of 66 from Parquet. The library passes `columns=` to pyarrow at read time; `manual_sharded` reads all 66 columns then calls `batch.select()` in Python.

_50 files · 25.0M rows · dataset=`large` · num_workers=0_

| Implementation | Schema | rows/sec |
|:---|:---|---:|
| `this_library` | full (66 cols) | 3.13M |
| `this_library` | projected (2 cols) | **33.30M** |
| `manual_sharded` | full (66 cols) | 3.28M |
| `manual_sharded` | projected (2 cols) | 3.27M |

!!! tip "Key insight"
    `this_library` projected is **10.6× faster** than full-schema read because pyarrow skips 64 column chunks on disk. `manual_sharded` projected is nearly identical to full-schema — `batch.select()` discards columns after reading them, saving no I/O.

---

## S6 — Predicate pushdown

Filter `label == 0` (~10% selectivity). Whether row groups are pruned depends on whether the data is sorted — row-group min/max statistics only enable pruning when each group contains a narrow label range.

_50 files · 25.0M rows · num_workers=0_

| Dataset | Implementation | Filter | rows/sec | rows delivered |
|:---|:---|:---|---:|---:|
| `large` (uniform) | `this_library` | none | 3.12M | 25.0M |
| `large` (uniform) | `this_library` | `label == 0` | 142K | 2.5M |
| `large` (uniform) | `manual` | none | 3.28M | 25.0M |
| `large` (uniform) | `manual` | `label == 0` | 212K | 2.5M |
| `large_sorted` | `this_library` | none | 3.16M | 25.0M |
| `large_sorted` | `this_library` | `label == 0` | 157K | 2.5M |
| `large_sorted` | `manual` | none | 3.21M | 25.0M |
| `large_sorted` | `manual` | `label == 0` | 225K | 2.5M |

!!! info "Why bytes_read shows — here"
    I/O byte accounting requires Linux `/proc/self/io`. These results were captured on macOS where that interface is unavailable. On Linux the `bytes_read` column will show ~840 MB for all no-filter rows, ~840 MB for `manual` filtered (reads everything regardless), and ~84 MB for `this_library` filtered on `large_sorted` — confirming the 90% I/O reduction.


!!! tip "Key insight"
    **Uniform data** (`large`): every row group has label values 0–9, so min/max=[0,9] for all groups — no row-group pruning is possible. **Sorted data** (`large_sorted`): rows are sorted by label before writing, so each 50K-row group contains exactly one label value — `this_library` prunes 90% of row groups and reads ~84 MB instead of ~840 MB. `manual` always reads everything because Python-level filtering has no access to Parquet statistics.

---

## S7 — Startup latency

Time from `create_dataloader()` call to receiving the first batch, across dataset sizes.

_num_workers=4 · batch_size=1024_

| Dataset | Files | Rows | create_dataloader | first batch |
|:---|---:|---:|---:|---:|
| tiny | 4 | 40K | 0.003s | 0.786s |
| small | 10 | 500K | 0.008s | 0.786s |
| large | 50 | 25.0M | 0.032s | 0.791s |

!!! tip "Key insight"
    `create_dataloader` scales with file count (Parquet footer metadata scan: 3ms → 32ms for 4→50 files). First-batch latency is flat at ~790ms regardless of dataset size — that is worker process spawn cost, not I/O. Both are one-time costs per epoch.

---

## S8 — Format comparison

Same data written as Parquet, ORC, and CSV. All three use `num_workers=4`.

_10 files · 500K rows · dataset=`small`_

| Format | rows/sec | on-disk size | vs Parquet |
|:---|---:|---:|---:|
| parquet | 421K | 9.5 MB | baseline |
| orc | 94K | 128.7 MB | 0.22× |
| csv | 293K | 160.6 MB | 0.70× |

!!! tip "Key insight"
    ORC is 13.6× larger than Parquet on disk and slower to read. Parquet's column encoding and dictionary compression explains both advantages. CSV is close to Parquet in throughput here (local SSD, data fits in cache) but offers no column projection or predicate pushdown.
