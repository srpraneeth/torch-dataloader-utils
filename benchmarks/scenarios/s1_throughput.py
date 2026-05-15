"""S1 — Baseline throughput.

Sweeps num_workers across [0, 2, 4] on equal-sized files (dataset: small).
Compares all three implementations. Primary comparison scenario.

Expected: this_library ≥ manual_sharded ≥ naive_iterable at every worker count.
naive_iterable degrades because each worker reads all files (I/O amplification).
"""

from __future__ import annotations

import benchmarks.baselines.manual_sharded as manual_sharded
import benchmarks.baselines.naive_iterable as naive_iterable
from benchmarks._common import load_manifest, measure, parquet_glob, passthrough
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = "Baseline throughput sweep across num_workers on equal-sized files. All three implementations."
DATASET = "small"
WORKER_COUNTS = [0, 2, 4, 8]
BATCH_SIZE = 1024


def run(data_dir: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    manifest = load_manifest(data_dir)
    expected = manifest["total_rows"]

    results: dict = {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "total_rows": expected,
        "num_files": len(manifest["files"]),
        "total_bytes": manifest["total_bytes"],
        "config": {"worker_counts": WORKER_COUNTS, "batch_size": BATCH_SIZE},
    }

    for nw in WORKER_COUNTS:
        key = f"num_workers={nw}"

        def _library(nw=nw):
            loader, _ = StructuredDataset.create_dataloader(
                path=parquet_glob(data_dir),
                format="parquet",
                num_workers=nw,
                batch_size=BATCH_SIZE,
                output_format="arrow",
                collate_fn=passthrough,
            )
            return loader

        def _manual(nw=nw):
            return manual_sharded.make_loader(data_dir, nw, BATCH_SIZE)

        def _naive(nw=nw):
            return naive_iterable.make_loader(data_dir, nw, BATCH_SIZE)

        results.setdefault("this_library", {})[key] = measure(_library, expected, n_warmup, n_runs)
        results.setdefault("manual_sharded", {})[key] = measure(_manual, expected, n_warmup, n_runs)
        results.setdefault("naive_iterable", {})[key] = measure(_naive, expected, n_warmup, n_runs)

    return results
