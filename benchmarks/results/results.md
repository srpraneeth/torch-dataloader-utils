# torch-dataloader-utils — benchmark results

| | |
|---|---|
| **Run at** | 2026-05-15T09:20:09 |
| **Git** | `a6994cc` |
| **Version** | v0.1.0 |
| **Platform** | darwin-arm64 |
| **Python** | 3.12.5 |

---

## Implementations

| Name | Description |
|:---|:---|
| `this_library` | torch-dataloader-utils (row-group-aware splits) |
| `manual_sharded` | Hand-written baseline: files pre-partitioned across workers at startup. No I/O waste, but can't split a single large file. |
| `naive_iterable` | Anti-pattern: every worker reads every file and discards non-assigned batches → N× I/O amplification. |

---

## Scorecard

| | Feature | this_library | vs baseline |
|:---|:---|:---|:---|
| **S1** | Throughput (nw=0) | **3.05M rows/s** | 1.01× vs naive_iter |
| **S2** | Load balance (unequal files) | **0.0% imbalance** | vs 116.7% manual  0.98× wall-clock |
| **S3** | Single large file (nw=8) | **0.51s epoch** | 5.87× faster than manual |
| **S4** | DDP I/O (16 ranks) | **1× per rank** | vs 16× naive_ddp |
| **S7** | Startup latency (50-file dataset) | **0.791s first batch** | metadata scan cost |
| **S8** | Format: parquet vs CSV | **1.44× faster** | 421K vs 293K rows/s |

## S1 — Baseline throughput sweep across num_workers on equal-sized files. All three implementations.  
_dataset=`small` · rows=25.0M · files=50_  
**config:** `worker_counts`=[0, 2, 4, 8] · `batch_size`=1024

| Implementation | num_workers=0 | num_workers=2 | num_workers=4 | num_workers=8 |
|:---|---:|---:|---:|---:|
| `this_library` | 3.05M | 2.13M | 1.98M | 1.72M |
| `manual_sharded` | 3.10M | 1.90M | 2.07M | 1.94M |
| `naive_iterable` | 3.02M | 1.53M | 1.29M | 789K |
| **Speedup (lib/naive)** | **1.01×** | **1.40×** | **1.53×** | **2.18×** |

> `naive_iterable` reads ALL files in every worker → throughput degrades with more workers.
> `naive_iterable` only appears in S1 — S2+ compare against `manual_sharded` only.

---

## S2 — Per-worker row distribution and wall-clock throughput on 32× unequal files. Imbalance from manual_sharded translates directly to slower epoch time.  
_dataset=`unequal` · rows=1.9M · files=20_  
**config:** `num_workers`=4 · `batch_size`=1024 · `split_rows`=2000

| Implementation | Workers | Total rows | Mean | Std Dev | Imbalance |
|:---|---:|---:|---:|---:|---:|
| `this_library` | 4 | 1.9M | 480K | 0 | 0.0% |
| `manual_sharded` | 4 | 1.9M | 480K | 206K | 116.7% |

**`this_library` per-worker:** ['480K', '480K', '480K', '480K']
**`manual_sharded` per-worker:** ['150K', '710K', '490K', '570K']

**Wall-clock throughput (num\_workers=4)**

| Implementation | rows/sec | elapsed | speedup |
|:---|---:|---:|---:|
| `this_library` | 980K | 1.959s | baseline |
| `manual_sharded` | 996K | 1.927s | **0.98×** |

> `manual_sharded` assigns whole files → large files land on one worker.
> `this_library` sub-splits files at row-group boundaries → near-equal distribution.
> `naive_iterable` excluded: every worker reads all files → 'perfect balance' but N× I/O waste.

---

## S3 — Single large file (10M rows). Sub-file splitting gives each worker 1/N of the file. Elapsed = max worker time (parallel simulation, no IPC overhead).  
_dataset=`single_large` · rows=10.0M · files=1_  
**config:** `num_workers`=8 · `batch_size`=1024 · `split_bytes`=10MiB

num\_workers=8

| Implementation | rows/sec (median) | elapsed (median) | speedup |
|:---|---:|---:|---:|
| `this_library` | 19.67M | 0.508s | baseline |
| `manual_sharded` | 3.35M | 2.983s | 0.17× |

| Implementation | batches/sec | GPU starves below |
|:---|---:|---:|
| `this_library` | 19,206 | 0.05ms |
| `manual_sharded` | 3,274 | 0.31ms |

> `manual_sharded` assigns the whole file to worker 0; workers 1–N are idle.
> `this_library` splits row groups across all workers → N× faster epoch time.
> **GPU starves below**: GPU steps faster than this threshold → data loader is the bottleneck.

---

## S4 — Rank-aware sharding: rank 0 of num_ranks receives total_rows/num_ranks. rows/sec stays constant — each rank does proportionally less work. naive_ddp reads ALL rows on every rank (N× I/O amplification).  
_dataset=`large` · rows=25.0M_  
**config:** `num_workers`=4 · `batch_size`=1024 · `rank_counts`=[1, 2, 4, 8, 16]


**`this_library` — reads 1/N of data**

| num_ranks | rows received | fraction | rows/sec | I/O amplification |
|---:|---:|---:|---:|---:|
| 1 | 25.0M | 100.0% | 2.08M | 1.0× |
| 2 | 12.5M | 50.0% | 1.83M | 1.0× |
| 4 | 6.5M | 26.0% | 1.71M | 1.0× |
| 8 | 3.5M | 14.0% | 1.40M | 1.0× |
| 16 | 2.0M | 8.0% | 1.11M | 1.0× |

**`naive_ddp` — reads ALL data, delivers 1/N**

| num_ranks | rows received | fraction | rows/sec | I/O amplification |
|---:|---:|---:|---:|---:|
| 1 | 25.0M | 100.0% | 2.12M | 1× |
| 2 | 12.5M | 50.0% | 1.06M | 2× |
| 4 | 6.5M | 26.0% | 551K | 4× |
| 8 | 3.5M | 14.0% | 297K | 8× |
| 16 | 2.0M | 8.0% | 170K | 16× |

> `naive_ddp` reads all files on every rank and discards non-assigned rows.
> At `num_ranks=8`, that's 8× more I/O for the same training data per rank.

---

## S5 — Column projection: full schema (66 cols) vs 2 cols. this_library skips 64 column chunks at I/O level; manual_sharded reads all then selects in Python.  
_dataset=`large` · rows=25.0M · files=50_  
**config:** `num_workers`=0 · `batch_size`=1024 · `projected_columns`=['row_id', 'label']

Projected columns: `['row_id', 'label']`

| Implementation | Schema | rows/sec | bytes read |
|:---|:---|---:|---:|
| `this_library` | full (66 cols) | 3.13M | — |
| `this_library` | projected (2 cols) | 33.30M | — |
| `manual_sharded` | full (66 cols) | 3.28M | — |
| `manual_sharded` | projected (2 cols) | 3.27M | — |

> `this_library` skips 64 column chunks at read time — true I/O reduction.
> `manual_sharded` reads all 66 columns then `batch.select(cols)` in Python — same bytes read.

---

## S6 — Predicate pushdown: label == 0 (~10% selectivity). Uniform data: no row-group pruning, all 4 impls read same bytes. Sorted data: this_library filtered prunes 90% of row groups, manual filtered still reads all bytes.  
_rows=25.0M_  
**config:** `num_workers`=0 · `batch_size`=1024 · `filter`=label == 0 · `selectivity`=~10%

Filter: `label == 0` (~10% selectivity)

| Dataset | Implementation | Filter | rows/sec | delivered | bytes read |
|:---|:---|:---|---:|---:|---:|
| `large` | `this_library` | `no filter` | 3.12M | 25.0M | — |
| `large` | `this_library` | `label == 0` | 142K | 2.5M | — |
| `large` | `manual` | `no filter` | 3.28M | 25.0M | — |
| `large` | `manual` | `label == 0` | 212K | 2.5M | — |
| `large_sorted` | `this_library` | `no filter` | 3.16M | 25.0M | — |
| `large_sorted` | `this_library` | `label == 0` | 157K | 2.5M | — |
| `large_sorted` | `manual` | `no filter` | 3.21M | 25.0M | — |
| `large_sorted` | `manual` | `label == 0` | 225K | 2.5M | — |

> `large`: label cycles 0–9 within every row group → min/max=[0,9] → no row groups pruned.
> `large_sorted`: rows sorted by label → each row group has one label → 90% of row groups pruned by `this_library`.
> `manual`: `iter_batches()` + `batch.filter()` in Python — reads all bytes regardless of sort order.

---

## S7 — Time from create_dataloader() to first batch across dataset sizes. Captures Parquet footer metadata scan cost.  
_rows=25.5M_  
**config:** `num_workers`=4 · `batch_size`=1024 · `datasets`=['tiny', 'small', 'large']

| Dataset | Files | Rows | create_dataloader (med) | time to first batch (med) |
|:---|---:|---:|---:|---:|
| tiny | 4 | 40K | 0.003s | 0.786s |
| small | 10 | 500K | 0.008s | 0.786s |
| large | 50 | 25.0M | 0.032s | 0.791s |

---

## S8 — Throughput comparison across Parquet, ORC, and CSV on equivalent data. Parquet and ORC benefit from sub-file splitting; CSV does not.  
_dataset=`small` · rows=500K_  
**config:** `num_workers`=4 · `batch_size`=1024 · `formats`=['parquet', 'orc', 'csv']

| Format | rows/sec | total bytes | vs parquet |
|:---|---:|---:|---:|
| parquet | 421K | 9.5 MB | baseline |
| orc | 94K | 128.7 MB | 0.22× |
| csv | 293K | 160.6 MB | 0.70× |

> ORC on-disk is 13.6× larger than Parquet — slower throughput AND larger storage.
> Parquet column encoding + dictionary compression explains the size and speed advantage.

---
