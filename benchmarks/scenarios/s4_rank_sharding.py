"""S4 — Rank-aware sharding.

Sweeps num_ranks across RANK_COUNTS with rank=0 fixed. Measures how many rows
rank 0 receives and how long one epoch takes. With correct interleaved sharding,
rows_received ≈ total_rows / num_ranks and throughput (rows/sec) stays constant.

Also measures naive_ddp: reads ALL rows without rank filtering, same as what
DistributedSampler does — every rank pays the full I/O cost, discarding N-1/N
of the data. I/O amplification = num_ranks×.
"""

from __future__ import annotations

from benchmarks._common import parquet_glob, passthrough, load_manifest, measure
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Rank-aware sharding: rank 0 of num_ranks receives total_rows/num_ranks. "
    "rows/sec stays constant — each rank does proportionally less work. "
    "naive_ddp reads ALL rows on every rank (N× I/O amplification)."
)
DATASET = "large"
NUM_WORKERS = 4
BATCH_SIZE = 1024
RANK_COUNTS = [1, 2, 4, 8, 16]


def run(data_dir: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    manifest = load_manifest(data_dir)
    total = manifest["total_rows"]

    results: dict = {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "total_rows": total,
        "num_workers": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "rank_counts": RANK_COUNTS},
        "this_library": {},
        "naive_ddp": {},
    }

    # naive_ddp baseline: one loader, no rank params — reads everything
    def _naive_loader():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir),
            format="parquet",
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            output_format="arrow",
            collate_fn=passthrough,
        )
        return loader

    naive_stats = measure(_naive_loader, total, n_warmup, n_runs)
    naive_stats["rows_received"] = total
    naive_stats["fraction_of_total"] = 1.0

    for nr in RANK_COUNTS:
        probe, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir),
            format="parquet",
            num_workers=0,
            batch_size=BATCH_SIZE,
            num_ranks=nr,
            rank=0,
            output_format="arrow",
            collate_fn=passthrough,
        )
        actual_rows = sum(b.num_rows for b in probe)

        def _loader(nr=nr):
            loader, _ = StructuredDataset.create_dataloader(
                path=parquet_glob(data_dir),
                format="parquet",
                num_workers=NUM_WORKERS,
                batch_size=BATCH_SIZE,
                num_ranks=nr,
                rank=0,
                output_format="arrow",
                collate_fn=passthrough,
            )
            return loader

        stats = measure(_loader, actual_rows, n_warmup, n_runs)
        stats["rows_received"] = actual_rows
        stats["fraction_of_total"] = round(actual_rows / total, 4)
        stats["io_amplification"] = 1.0  # reads only its share
        results["this_library"][f"num_ranks={nr}"] = stats

        # naive_ddp at this num_ranks: reads all rows but delivers only 1/nr fraction
        naive_elapsed = naive_stats["elapsed_sec"]["median"]
        naive_rps = naive_stats["rows_per_sec"]["median"]
        results["naive_ddp"][f"num_ranks={nr}"] = {
            "rows_received": actual_rows,
            "fraction_of_total": round(actual_rows / total, 4),
            "elapsed_sec": naive_stats["elapsed_sec"],
            "rows_per_sec": {"median": round(actual_rows / naive_elapsed)},
            "io_amplification": nr,
        }

    return results
