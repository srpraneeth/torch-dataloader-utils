"""S3 — Single large file with sub-file splitting.

Simulates NUM_WORKERS parallel workers using rank simulation (num_workers=0 per
loader) to eliminate DataLoader IPC overhead — same technique as S2.

Wall-clock model: workers run in parallel → elapsed = max(per-worker time).
  this_library   → splits file into NUM_WORKERS chunks; each reads 1/N
  manual_sharded → worker 0 reads everything; workers 1–N−1 return immediately
"""

from __future__ import annotations

import statistics
import time

from benchmarks._common import parquet_glob, passthrough, load_manifest, run_epoch
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Single large file (10M rows). Sub-file splitting gives each worker 1/N of "
    "the file. Elapsed = max worker time (parallel simulation, no IPC overhead)."
)
DATASET = "single_large"
NUM_WORKERS = 8
BATCH_SIZE = 1024
SPLIT_BYTES = 10 * 1024 * 1024  # 10 MiB — ~26 chunks across 8 workers


def _simulate_parallel(make_loader_fn, num_workers: int, n_runs: int) -> dict:
    """Rank simulation: run each worker sequentially, take max as wall-clock."""
    all_max_elapsed = []
    total_rows = 0

    for _ in range(n_runs):
        worker_times = []
        run_rows = 0
        for w in range(num_workers):
            loader = make_loader_fn(w)
            t0 = time.perf_counter()
            rows, _ = run_epoch(loader)
            worker_times.append(time.perf_counter() - t0)
            run_rows += rows
        all_max_elapsed.append(max(worker_times))
        total_rows = run_rows

    med = statistics.median(all_max_elapsed)
    batches_per_sec = round(total_rows / BATCH_SIZE / med)
    return {
        "elapsed_sec": {"median": round(med, 4), "min": round(min(all_max_elapsed), 4)},
        "rows_per_sec": {"median": round(total_rows / med)},
        "batches_per_sec": batches_per_sec,
        "total_rows": total_rows,
    }


def run(data_dir: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    manifest = load_manifest(data_dir)
    expected = manifest["total_rows"]

    # Warmup: prime OS disk cache
    for _ in range(n_warmup):
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet", num_workers=0,
            batch_size=BATCH_SIZE, split_bytes=SPLIT_BYTES,
            output_format="arrow", collate_fn=passthrough,
        )
        run_epoch(loader)

    def _lib_worker(w: int):
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet", num_workers=0,
            batch_size=BATCH_SIZE, split_bytes=SPLIT_BYTES,
            num_ranks=NUM_WORKERS, rank=w,
            output_format="arrow", collate_fn=passthrough,
        )
        return loader

    def _manual_worker(w: int):
        from glob import glob
        import pyarrow.parquet as pq
        files = sorted(glob(f"{data_dir}/*.parquet"))
        my_files = files[w::NUM_WORKERS]

        class _W:
            def __iter__(self_):
                for path in my_files:
                    pf = pq.ParquetFile(path)
                    for batch in pf.iter_batches(BATCH_SIZE):
                        if batch.num_rows > 0:
                            yield batch
        return _W()

    lib_stats = _simulate_parallel(_lib_worker, NUM_WORKERS, n_runs)
    manual_stats = _simulate_parallel(_manual_worker, NUM_WORKERS, n_runs)

    return {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "total_rows": expected,
        "num_files": len(manifest["files"]),
        "num_workers": NUM_WORKERS,
        "config": {
            "num_workers": NUM_WORKERS,
            "batch_size": BATCH_SIZE,
            "split_bytes": f"{SPLIT_BYTES // (1024 * 1024)}MiB",
        },
        "this_library": lib_stats,
        "manual_sharded": manual_stats,
    }
