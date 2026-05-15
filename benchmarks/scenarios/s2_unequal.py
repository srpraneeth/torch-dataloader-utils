"""S2 — Load balance with unequal file sizes.

Uses the 'unequal' dataset (20 files, geometric size distribution 10K–320K rows, 32× ratio).
Measures two things:
  1. Per-worker row distribution — shows imbalance directly
  2. Wall-clock throughput with num_workers=4 — shows imbalance costs real time

Expected:
  this_library  → ~2% imbalance (sub-file splitting equalises chunks)
  manual_sharded → ~100%+ imbalance (whole-file round-robin; large files dominate one worker)
"""

from __future__ import annotations

import statistics

import benchmarks.baselines.manual_sharded as manual_sharded_baseline
from benchmarks._common import load_manifest, measure, parquet_glob, passthrough, run_epoch
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Per-worker row distribution and wall-clock throughput on 32× unequal files. "
    "Imbalance from manual_sharded translates directly to slower epoch time."
)
DATASET = "unequal"
NUM_WORKERS = 4
BATCH_SIZE = 1024
SPLIT_ROWS = 2_000


def _simulate_workers_library(data_dir: str, num_workers: int) -> list[int]:
    counts = []
    for w in range(num_workers):
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir),
            format="parquet",
            num_workers=0,
            batch_size=BATCH_SIZE,
            num_ranks=num_workers,
            rank=w,
            split_rows=SPLIT_ROWS,
            output_format="arrow",
            collate_fn=passthrough,
        )
        rows, _ = run_epoch(loader)
        counts.append(rows)
    return counts


def _simulate_workers_manual(data_dir: str, num_workers: int) -> list[int]:
    from glob import glob

    import pyarrow.parquet as pq

    files = sorted(glob(f"{data_dir}/*.parquet"))
    counts = []
    for w in range(num_workers):
        my_files = files[w::num_workers]
        rows = sum(pq.read_metadata(f).num_rows for f in my_files)
        counts.append(rows)
    return counts


def _balance_stats(counts: list[int]) -> dict:
    total = sum(counts)
    mean = statistics.mean(counts)
    std = statistics.pstdev(counts)
    return {
        "per_worker_rows": counts,
        "total_rows": total,
        "mean": round(mean),
        "std_dev": round(std),
        "max": max(counts),
        "min": min(counts),
        "imbalance_pct": round(100 * (max(counts) - min(counts)) / mean, 1),
    }


def run(data_dir: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    manifest = load_manifest(data_dir)
    total = manifest["total_rows"]

    library_counts = _simulate_workers_library(data_dir, NUM_WORKERS)
    manual_counts = _simulate_workers_manual(data_dir, NUM_WORKERS)

    # Wall-clock throughput: imbalance means the slowest worker is the bottleneck.
    # With num_workers=4, all workers run in parallel — total time = max worker time.
    def _lib_loader():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir),
            format="parquet",
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            split_rows=SPLIT_ROWS,
            output_format="arrow",
            collate_fn=passthrough,
        )
        return loader

    def _manual_loader():
        return manual_sharded_baseline.make_loader(data_dir, NUM_WORKERS, BATCH_SIZE)

    lib_throughput = measure(_lib_loader, total, n_warmup, n_runs)
    manual_throughput = measure(_manual_loader, total, n_warmup, n_runs)

    return {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "total_rows": total,
        "num_files": len(manifest["files"]),
        "num_workers_simulated": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "split_rows": SPLIT_ROWS},
        "this_library": {**_balance_stats(library_counts), "throughput": lib_throughput},
        "manual_sharded": {**_balance_stats(manual_counts), "throughput": manual_throughput},
    }
