"""S8 — Format comparison: Parquet vs ORC vs CSV.

Compares throughput of this_library across three formats on equivalent data.
Uses num_workers=4 with dataset 'small'.

Expected: Parquet ≥ ORC > CSV (sub-file splitting available for Parquet and ORC;
CSV is whole-file only so parallelism is file-count limited).
"""

from __future__ import annotations

import os

from benchmarks._common import format_glob, passthrough, load_manifest, measure
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Throughput comparison across Parquet, ORC, and CSV on equivalent data. "
    "Parquet and ORC benefit from sub-file splitting; CSV does not."
)
DATASET = "small"
NUM_WORKERS = 4
BATCH_SIZE = 1024

FORMAT_DIRS = {
    "parquet": "",          # {root}/small
    "orc": "_orc",          # {root}/small_orc
    "csv": "_csv",          # {root}/small_csv
}


def run(data_dir_root: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    results: dict = {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "num_workers": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "formats": list(FORMAT_DIRS.keys())},
        "this_library": {},
    }

    total_rows = 0
    for fmt, suffix in FORMAT_DIRS.items():
        data_dir = os.path.join(data_dir_root, f"{DATASET}{suffix}")
        if not os.path.isdir(data_dir):
            results["this_library"][fmt] = {"skipped": f"{data_dir} not found — run gen_data.py --format {fmt}"}
            continue

        manifest = load_manifest(data_dir)
        expected = manifest["total_rows"]
        if not total_rows:
            total_rows = expected

        def _loader(fmt=fmt, data_dir=data_dir):
            loader, _ = StructuredDataset.create_dataloader(
                path=format_glob(data_dir, fmt),
                format=fmt,
                num_workers=NUM_WORKERS,
                batch_size=BATCH_SIZE,
                output_format="arrow",
                collate_fn=passthrough,
            )
            return loader

        stats = measure(_loader, expected, n_warmup, n_runs)
        stats["total_bytes"] = manifest["total_bytes"]
        results["this_library"][fmt] = stats

    results["total_rows"] = total_rows
    return results
