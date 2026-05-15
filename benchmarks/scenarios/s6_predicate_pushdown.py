"""S6 — Predicate pushdown selectivity.

Runs the same filter (label == 0, ~10% selectivity) on two datasets:
  - large:        label cycles 0-9 within every row group → min/max=[0,9] for
                  every group → no row groups can be pruned → bytes_read ≈ full
  - large_sorted: rows sorted by label → each row group contains one label value
                  → filter prunes 90% of row groups → bytes_read ≈ 10% of full

Four measurements per dataset:
  this_library/no_filter   — StructuredDataset, no filters= arg
  this_library/filtered    — StructuredDataset, filters=FILTER (row-group pushdown)
  manual/no_filter         — plain pq.ParquetFile.iter_batches, no filter
  manual/filtered          — plain pq.ParquetFile.iter_batches, then batch.filter() in Python
"""

from __future__ import annotations

import glob as _glob
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from benchmarks._common import parquet_glob, passthrough, load_manifest, measure, measure_io_bytes, run_epoch
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Predicate pushdown: label == 0 (~10% selectivity). "
    "Uniform data: no row-group pruning, all 4 impls read same bytes. "
    "Sorted data: this_library filtered prunes 90% of row groups, manual filtered still reads all bytes."
)
NUM_WORKERS = 0
BATCH_SIZE = 1024
FILTER = pc.field("label") == 0


def _manual_epoch(data_dir: str, apply_filter: bool) -> pa.RecordBatch:
    """Iterate files with plain pq.ParquetFile.iter_batches; optionally filter in Python."""
    for path in sorted(_glob.glob(parquet_glob(data_dir))):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(BATCH_SIZE):
            if apply_filter:
                mask = pc.equal(batch.column("label"), 0)
                batch = batch.filter(mask)
            yield batch


def _measure_dataset(data_dir: str, n_warmup: int, n_runs: int) -> dict:
    manifest = load_manifest(data_dir)
    total_rows = manifest["total_rows"]

    # --- count filtered rows once ---
    filtered_rows = sum(
        b.num_rows for b in _manual_epoch(data_dir, apply_filter=True)
    )

    # --- this_library loaders ---
    def _lib_no_filter():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet",
            num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
            output_format="arrow", collate_fn=passthrough,
        )
        return loader

    def _lib_filtered():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet",
            num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
            filters=FILTER, output_format="arrow", collate_fn=passthrough,
        )
        return loader

    # --- manual loaders ---
    def _manual_no_filter():
        return _manual_epoch(data_dir, apply_filter=False)

    def _manual_filtered():
        return _manual_epoch(data_dir, apply_filter=True)

    # --- I/O bytes (single pass each) ---
    _, lib_nf_bytes   = measure_io_bytes(lambda: run_epoch(_lib_no_filter()))
    _, lib_f_bytes    = measure_io_bytes(lambda: run_epoch(_lib_filtered()))
    _, man_nf_bytes   = measure_io_bytes(lambda: run_epoch(_manual_no_filter()))
    _, man_f_bytes    = measure_io_bytes(lambda: run_epoch(_manual_filtered()))

    # --- throughput ---
    lib_nf_stats  = measure(_lib_no_filter,  total_rows,    n_warmup, n_runs)
    lib_f_stats   = measure(_lib_filtered,   filtered_rows, n_warmup, n_runs)
    man_nf_stats  = measure(_manual_no_filter,  total_rows,    n_warmup, n_runs)
    man_f_stats   = measure(_manual_filtered,   filtered_rows, n_warmup, n_runs)

    # --- annotate ---
    for stats, rows in [
        (lib_nf_stats, total_rows), (lib_f_stats, filtered_rows),
        (man_nf_stats, total_rows), (man_f_stats, filtered_rows),
    ]:
        stats["total_rows"] = rows

    lib_f_stats["selectivity_pct"] = round(100 * filtered_rows / total_rows, 1)
    man_f_stats["selectivity_pct"] = round(100 * filtered_rows / total_rows, 1)

    if lib_nf_bytes:
        lib_nf_stats["bytes_read"] = lib_nf_bytes
    if lib_f_bytes:
        lib_f_stats["bytes_read"] = lib_f_bytes
        if lib_nf_bytes:
            lib_f_stats["bytes_reduction_pct"] = round(100 * (1 - lib_f_bytes / lib_nf_bytes), 1)
    if man_nf_bytes:
        man_nf_stats["bytes_read"] = man_nf_bytes
    if man_f_bytes:
        man_f_stats["bytes_read"] = man_f_bytes
        if man_nf_bytes:
            man_f_stats["bytes_reduction_pct"] = round(100 * (1 - man_f_bytes / man_nf_bytes), 1)

    return {
        "num_files": len(manifest["files"]),
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "this_library": {
            "no_filter": lib_nf_stats,
            "filtered":  lib_f_stats,
        },
        "manual": {
            "no_filter": man_nf_stats,
            "filtered":  man_f_stats,
        },
    }


def run(data_dir_root: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    results: dict = {
        "description": DESCRIPTION,
        "num_workers": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "filter": "label == 0", "selectivity": "~10%"},
        "datasets": {},
    }
    total_rows = 0

    for ds_name in ["large", "large_sorted"]:
        data_dir = os.path.join(data_dir_root, ds_name)
        if not os.path.isdir(data_dir):
            results["datasets"][ds_name] = {"skipped": f"{data_dir} not found — run gen_data.py --dataset {ds_name}"}
            continue
        stats = _measure_dataset(data_dir, n_warmup, n_runs)
        results["datasets"][ds_name] = stats
        if not total_rows:
            total_rows = stats["total_rows"]

    results["total_rows"] = total_rows
    return results
