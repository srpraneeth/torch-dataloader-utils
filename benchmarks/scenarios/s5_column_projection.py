"""S5 — Column projection speedup.

Compares full-schema reads vs projecting down to 2 of 66 columns, for both
this_library and manual_sharded.

  this_library (projected)  → passes columns= to pyarrow scanner → reads only
                               those column chunks from disk (true I/O reduction)
  manual_sharded (projected) → reads all 66 columns from disk, then selects 2
                               in Python (batch.select) — same I/O as full read

This shows that the speedup comes from I/O skipping, not in-memory filtering.
"""

from __future__ import annotations

from glob import glob

import pyarrow.parquet as pq

from benchmarks._common import load_manifest, measure, measure_io_bytes, parquet_glob, passthrough, run_epoch
from torch_dataloader_utils import StructuredDataset

DESCRIPTION = (
    "Column projection: full schema (66 cols) vs 2 cols. "
    "this_library skips 64 column chunks at I/O level; manual_sharded reads all then selects in Python."
)
DATASET = "large"
NUM_WORKERS = 0  # single-process: isolates I/O projection effect from IPC overhead
BATCH_SIZE = 1024
PROJECTED_COLS = ["row_id", "label"]


def _manual_loader(data_dir: str, select_cols=None):
    """Reads all columns from disk, then selects in Python — no I/O pushdown."""
    files = sorted(glob(f"{data_dir}/*.parquet"))

    class _W:
        def __iter__(self_):
            for path in files:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(BATCH_SIZE):
                    if select_cols is not None:
                        batch = batch.select(select_cols)
                    if batch.num_rows > 0:
                        yield batch

    return _W()


def run(data_dir: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    manifest = load_manifest(data_dir)
    expected = manifest["total_rows"]
    total_bytes = manifest["total_bytes"]

    # --- this_library loaders ---
    def _lib_full():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet",
            num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
            output_format="arrow", collate_fn=passthrough,
        )
        return loader

    def _lib_projected():
        loader, _ = StructuredDataset.create_dataloader(
            path=parquet_glob(data_dir), format="parquet",
            num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
            columns=PROJECTED_COLS, output_format="arrow", collate_fn=passthrough,
        )
        return loader

    # --- manual_sharded loaders ---
    def _manual_full():
        return _manual_loader(data_dir, select_cols=None)

    def _manual_select():
        return _manual_loader(data_dir, select_cols=PROJECTED_COLS)

    # --- I/O bytes (num_workers=0 already, so measure during throughput runs) ---
    _, lib_full_bytes = measure_io_bytes(lambda: run_epoch(_lib_full()))
    _, lib_proj_bytes = measure_io_bytes(lambda: run_epoch(_lib_projected()))
    _, man_full_bytes = measure_io_bytes(lambda: run_epoch(_manual_full()))
    _, man_sel_bytes  = measure_io_bytes(lambda: run_epoch(_manual_select()))

    # --- throughput ---
    lib_full_stats = measure(_lib_full, expected, n_warmup, n_runs)
    lib_proj_stats = measure(_lib_projected, expected, n_warmup, n_runs)
    man_full_stats = measure(_manual_full, expected, n_warmup, n_runs)
    man_sel_stats = measure(_manual_select, expected, n_warmup, n_runs)

    def _attach_bytes(stats, b_read, b_full):
        if b_read:
            stats["bytes_read"] = b_read
        if b_read and b_full:
            stats["bytes_reduction_pct"] = round(100 * (1 - b_read / b_full), 1)

    _attach_bytes(lib_proj_stats, lib_proj_bytes, lib_full_bytes)
    _attach_bytes(man_sel_stats, man_sel_bytes, man_full_bytes)
    if lib_full_bytes:
        lib_full_stats["bytes_read"] = lib_full_bytes
    if man_full_bytes:
        man_full_stats["bytes_read"] = man_full_bytes

    return {
        "description": DESCRIPTION,
        "dataset": DATASET,
        "total_rows": expected,
        "num_files": len(manifest["files"]),
        "num_workers": NUM_WORKERS,
        "config": {"num_workers": NUM_WORKERS, "batch_size": BATCH_SIZE, "projected_columns": PROJECTED_COLS},
        "projected_columns": PROJECTED_COLS,
        "this_library": {
            "full_schema": lib_full_stats,
            "projected": lib_proj_stats,
        },
        "manual_sharded": {
            "full_schema": man_full_stats,
            "projected": man_sel_stats,
        },
    }

