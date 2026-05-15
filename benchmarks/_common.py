"""Shared measurement utilities for all benchmark scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from typing import Callable

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ---------------------------------------------------------------------------
# Collate function — must be module-level so it is picklable by DataLoader workers
# ---------------------------------------------------------------------------


def passthrough(x):
    """Identity collate_fn: pass Arrow RecordBatch through without modification."""
    return x


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------


def run_epoch(loader) -> tuple[int, float]:
    """Iterate one full epoch. Returns (total_rows, elapsed_seconds)."""
    t0 = time.perf_counter()
    total_rows = 0
    for batch in loader:
        if hasattr(batch, "num_rows"):  # pyarrow.RecordBatch
            total_rows += batch.num_rows
        elif isinstance(batch, dict):
            total_rows += len(next(iter(batch.values())))
        else:
            total_rows += len(batch)
    return total_rows, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def measure(
    make_loader: Callable,
    expected_rows: int,
    n_warmup: int = 1,
    n_runs: int = 5,
) -> dict:
    """Time make_loader() over n_runs epochs. Returns a stats dict.

    make_loader is called fresh for each run so DataLoader workers are
    re-spawned and startup overhead is included in each measurement.
    The first n_warmup calls are discarded (warm OS page cache, JIT).
    """
    for _ in range(n_warmup):
        loader = make_loader()
        actual, _ = run_epoch(loader)
        if actual != expected_rows:
            raise AssertionError(f"Warm-up row mismatch: got {actual}, expected {expected_rows}")

    times: list[float] = []
    for _ in range(n_runs):
        loader = make_loader()
        actual, elapsed = run_epoch(loader)
        if actual != expected_rows:
            raise AssertionError(f"Row mismatch: got {actual}, expected {expected_rows}")
        times.append(elapsed)

    return _stats(times, expected_rows)


def _stats(times: list[float], total_rows: int) -> dict:
    s = sorted(times)
    n = len(s)
    median = statistics.median(s)
    p25 = s[n // 4]
    p75 = s[(3 * n) // 4]
    return {
        "elapsed_sec": {
            "median": round(median, 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "min": round(s[0], 4),
        },
        # note: throughput p25/p75 are inverted from time p25/p75
        "rows_per_sec": {
            "median": round(total_rows / median),
            "p25": round(total_rows / p75),
            "p75": round(total_rows / p25),
        },
        "total_rows": total_rows,
    }


# ---------------------------------------------------------------------------
# I/O byte measurement (single-process only, requires psutil)
# ---------------------------------------------------------------------------


def measure_io_bytes(fn: Callable) -> tuple[any, int | None]:
    """Call fn() and return (result, bytes_read). bytes_read is None if psutil
    is unavailable or the platform does not support io_counters."""
    if not _PSUTIL:
        return fn(), None
    proc = psutil.Process()
    try:
        before = proc.io_counters()
        result = fn()
        after = proc.io_counters()
        return result, after.read_bytes - before.read_bytes
    except (AttributeError, psutil.AccessDenied):
        # macOS may not expose read_bytes in all configurations
        return fn(), None


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No manifest.json found in {data_dir}\n"
            f"Run: python -m benchmarks.gen_data --out-dir <parent> --dataset <name>"
        )
    with open(path) as f:
        return json.load(f)


def verify_manifest(data_dir: str) -> dict:
    """Load and checksum-verify all files listed in manifest.json."""
    manifest = load_manifest(data_dir)
    for entry in manifest["files"]:
        fpath = os.path.join(data_dir, entry["name"])
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Benchmark file missing: {fpath}")
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {entry['name']} — re-run gen_data.py"
            )
    return manifest


def parquet_glob(data_dir: str) -> str:
    """Return a glob pattern for parquet files, excluding manifest.json."""
    return os.path.join(data_dir, "*.parquet")


def format_glob(data_dir: str, fmt: str) -> str:
    return os.path.join(data_dir, f"*.{fmt}")
