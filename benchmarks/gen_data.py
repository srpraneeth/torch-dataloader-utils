#!/usr/bin/env python3
"""Generate synthetic benchmark datasets.

Each dataset is written to <out_dir>/<dataset_name>/ (parquet) and optionally
<out_dir>/<dataset_name>_orc/ or <out_dir>/<dataset_name>_csv/ for S8.

A manifest.json is written alongside the files containing file names, sizes,
row counts, and SHA-256 checksums. Benchmarks verify the manifest before
measuring to ensure data has not changed between runs.

Usage:
    python benchmarks/gen_data.py --out-dir /tmp/bench_data --dataset small
    python benchmarks/gen_data.py --out-dir /tmp/bench_data --dataset all
    python benchmarks/gen_data.py --out-dir /tmp/bench_data --dataset small --format all
"""

import argparse
import hashlib
import json
import os
import random as _random

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.orc as pa_orc
import pyarrow.parquet as pq

BENCH_SEED = 42

# fmt: off
DATASETS: dict[str, dict] = {
    "tiny":         {"files": 4,  "base_rows": 10_000,    "row_group": 2_000,  "unequal": False, "sorted_by_label": False},
    "small":        {"files": 10, "base_rows": 50_000,    "row_group": 5_000,  "unequal": False, "sorted_by_label": False},
    "large":        {"files": 50, "base_rows": 500_000,   "row_group": 50_000, "unequal": False, "sorted_by_label": False},
    "large_sorted": {"files": 50, "base_rows": 500_000,   "row_group": 50_000, "unequal": False, "sorted_by_label": True},
    "unequal":      {"files": 20, "base_rows": 10_000,    "row_group": 5_000,  "unequal": True,  "sorted_by_label": False},
    "single_large": {"files": 1,  "base_rows": 10_000_000, "row_group": 50_000, "unequal": False, "sorted_by_label": False},
}
# fmt: on


def _schema() -> pa.Schema:
    fields = [pa.field("row_id", pa.int32())]
    for i in range(64):
        fields.append(pa.field(f"feat_{i:02d}", pa.float32()))
    fields.append(pa.field("label", pa.int32()))
    return pa.schema(fields)


def make_table(n_rows: int, row_id_offset: int) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    cols: dict[str, pa.Array] = {"row_id": pa.array(row_ids, type=pa.int32())}
    for i in range(64):
        cols[f"feat_{i:02d}"] = pa.array(
            [float((r + i) % 100) / 100.0 for r in row_ids], type=pa.float32()
        )
    cols["label"] = pa.array([r % 10 for r in row_ids], type=pa.int32())
    return pa.table(cols, schema=_schema())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(out_dir: str, dataset: str, fmt: str = "parquet") -> dict:
    """Generate one dataset in the given format. Returns the manifest dict."""
    cfg = DATASETS[dataset]
    os.makedirs(out_dir, exist_ok=True)

    manifest: dict = {
        "dataset": dataset,
        "format": fmt,
        "seed": BENCH_SEED,
        "files": [],
        "total_rows": 0,
    }
    total = 0

    # For unequal datasets, pre-compute shuffled sizes so large files can
    # cluster arbitrarily — round-robin assignment then shows realistic worst-case
    # imbalance instead of accidentally mixing sizes due to cyclic patterns.
    if cfg["unequal"]:
        sizes = [cfg["base_rows"] * (2 ** (j % 6)) for j in range(cfg["files"])]
        _random.Random(BENCH_SEED).shuffle(sizes)
    else:
        sizes = [cfg["base_rows"]] * cfg["files"]

    for i in range(cfg["files"]):
        n_rows = sizes[i]

        table = make_table(n_rows, row_id_offset=total)

        if cfg.get("sorted_by_label"):
            import pyarrow.compute as pc
            table = table.sort_by("label")

        if fmt == "parquet":
            path = os.path.join(out_dir, f"part_{i:04d}.parquet")
            pq.write_table(table, path, row_group_size=cfg["row_group"])
        elif fmt == "orc":
            path = os.path.join(out_dir, f"part_{i:04d}.orc")
            pa_orc.write_table(table, path)
        elif fmt == "csv":
            path = os.path.join(out_dir, f"part_{i:04d}.csv")
            pa_csv.write_csv(table, path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        size = os.path.getsize(path)
        manifest["files"].append(
            {
                "name": os.path.basename(path),
                "rows": n_rows,
                "size": size,
                "sha256": _sha256(path),
            }
        )
        total += n_rows
        print(f"  {os.path.basename(path):30s}  rows={n_rows:>9,}  size={size:>12,} B")

    manifest["total_rows"] = total
    manifest["total_bytes"] = sum(e["size"] for e in manifest["files"])

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total: {total:,} rows  {manifest['total_bytes']:,} bytes → {manifest_path}\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument("--out-dir", required=True, help="Root output directory")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="small",
        help="Dataset size profile (default: small)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "orc", "csv", "all"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    args = parser.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    formats = ["parquet", "orc", "csv"] if args.format == "all" else [args.format]

    for ds in datasets:
        for fmt in formats:
            suffix = "" if fmt == "parquet" else f"_{fmt}"
            out = os.path.join(args.out_dir, f"{ds}{suffix}")
            print(f"=== {ds} ({fmt}) → {out} ===")
            generate(out, ds, fmt)


if __name__ == "__main__":
    main()
