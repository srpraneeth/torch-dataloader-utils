"""S7 — Startup latency.

Measures time from create_dataloader() call to receipt of first batch,
across dataset sizes (tiny, small, medium, large). This captures the
metadata scan cost — reading Parquet footers to determine row group sizes.

The tradeoff: startup latency grows with file count (O(files)), but enables
sub-file splitting and exact load balancing. manual_sharded has near-zero
startup but cannot split files.
"""

from __future__ import annotations

import os
import time

from benchmarks._common import parquet_glob, passthrough, load_manifest
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Time from create_dataloader() to first batch across dataset sizes. "
    "Captures Parquet footer metadata scan cost."
)
DATASETS = ["tiny", "small", "large"]
NUM_WORKERS = 4
BATCH_SIZE = 1024
N_RUNS = 5


def _measure_startup(data_dir: str) -> dict:
    times_create = []
    times_first_batch = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir),
            format="parquet",
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            output_format="arrow",
            collate_fn=passthrough,
        )
        t_create = time.perf_counter() - t0
        times_create.append(t_create)

        t1 = time.perf_counter()
        it = iter(loader)
        next(it)
        t_first = time.perf_counter() - t1
        times_first_batch.append(t_first)

        # drain the rest to avoid worker leaks
        for _ in it:
            pass

    import statistics

    return {
        "create_dataloader_sec": {
            "median": round(statistics.median(times_create), 4),
            "min": round(min(times_create), 4),
        },
        "time_to_first_batch_sec": {
            "median": round(statistics.median(times_first_batch), 4),
            "min": round(min(times_first_batch), 4),
        },
    }


def run(data_dir_root: str, **_kwargs) -> dict:
    """data_dir_root is the parent directory containing <dataset>/ subdirs."""
    results: dict = {
        "description": DESCRIPTION,
        "num_workers": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "datasets": DATASETS},
        "this_library": {},
    }

    for ds in DATASETS:
        data_dir = os.path.join(data_dir_root, ds)
        if not os.path.isdir(data_dir):
            results["this_library"][ds] = {"skipped": f"{data_dir} not found"}
            continue
        manifest = load_manifest(data_dir)
        stats = _measure_startup(data_dir)
        stats["num_files"] = len(manifest["files"])
        stats["total_rows"] = manifest["total_rows"]
        results["this_library"][ds] = stats

    results["total_rows"] = sum(
        v.get("total_rows", 0)
        for v in results["this_library"].values()
        if isinstance(v, dict) and "skipped" not in v
    )
    return results
