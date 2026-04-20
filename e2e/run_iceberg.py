"""
E2E pipeline test: IcebergDataset → DataLoader → correctness checks.

Creates a local SQLite-backed Iceberg catalog, populates a table with
synthetic data across multiple files, then validates:
  - All rows returned, no duplicates, no gaps
  - Column projection
  - Predicate filters
  - Multi-epoch shuffle
  - Multi-worker disjoint reads

Requirements:
    pip install torch-dataloader-utils[iceberg]

Usage:
    uv run python e2e/run_iceberg.py
    uv run python e2e/run_iceberg.py --num-files 6 --rows 5000 --num-workers 4
    uv run python e2e/run_iceberg.py --no-shuffle
"""
import argparse
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.compute as pc

try:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import FloatType, IntegerType, NestedField, StringType
except ImportError:
    print("ERROR: pyiceberg not installed. Run: pip install torch-dataloader-utils[iceberg]")
    sys.exit(1)

import torch
from torch_dataloader_utils.dataset.iceberg import IcebergDataset


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _iceberg_schema() -> Schema:
    return Schema(
        NestedField(1, "row_id",    IntegerType(), required=False),
        NestedField(2, "feature_a", FloatType(),   required=False),
        NestedField(3, "feature_b", IntegerType(), required=False),
        NestedField(4, "label",     IntegerType(), required=False),
    )


def _make_arrow_batch(n_rows: int, offset: int) -> pa.Table:
    row_ids = list(range(offset, offset + n_rows))
    return pa.table({
        "row_id":    pa.array(row_ids,                                  type=pa.int32()),
        "feature_a": pa.array([float(i % 10) / 10.0 for i in row_ids], type=pa.float32()),
        "feature_b": pa.array([i % 100 for i in row_ids],              type=pa.int32()),
        "label":     pa.array([i % 2 for i in row_ids],                type=pa.int32()),
    })


def setup_catalog(warehouse_dir: str, db_path: str) -> tuple[SqlCatalog, dict]:
    catalog = SqlCatalog(
        "e2e",
        **{
            "uri": f"sqlite:///{db_path}",
            "warehouse": f"file://{warehouse_dir}",
        },
    )
    catalog.create_namespace("e2e")
    config = {
        "name": "e2e",
        "uri": f"sqlite:///{db_path}",
        "warehouse": f"file://{warehouse_dir}",
    }
    return catalog, config


def create_table(catalog: SqlCatalog, num_files: int, rows_per_file: int) -> int:
    """Create a table and append num_files batches. Returns total row count."""
    table = catalog.create_table("e2e.test_table", schema=_iceberg_schema())
    total = 0
    for i in range(num_files):
        batch = _make_arrow_batch(rows_per_file, offset=total)
        table.append(batch)
        total += rows_per_file
    return total


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect(it) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in it:
        for key, val in batch.items():
            result.setdefault(key, [])
            if isinstance(val, torch.Tensor):
                result[key].extend(val.tolist())
            elif hasattr(val, "tolist"):
                result[key].extend(val.tolist())
            else:
                result[key].extend(list(val))
    return result


# ---------------------------------------------------------------------------
# Individual scenario checks
# ---------------------------------------------------------------------------

def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✓" if condition else "✗ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def run_basic_read(catalog_config: dict, total_rows: int, batch_size: int) -> bool:
    print("\n[1] Basic read — all rows, no duplicates, no gaps")
    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        batch_size=batch_size,
    )
    rows = collect(loader)
    row_ids = sorted(rows.get("row_id", []))
    ok_count = check("row count", len(row_ids) == total_rows,
                     f"got {len(row_ids)}, expected {total_rows}")
    ok_dedup = check("no duplicates", len(row_ids) == len(set(row_ids)))
    ok_range = check("no gaps", row_ids == list(range(total_rows)))
    return ok_count and ok_dedup and ok_range


def run_column_projection(catalog_config: dict) -> bool:
    print("\n[2] Column projection — only feature_a and label returned")
    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        columns=["feature_a", "label"],
    )
    batch = next(iter(loader))
    ok_present = check("projected columns present", set(batch.keys()) == {"feature_a", "label"})
    ok_absent = check("extra columns absent", "row_id" not in batch and "feature_b" not in batch)
    return ok_present and ok_absent


def run_predicate_filter(catalog_config: dict, total_rows: int) -> bool:
    print("\n[3] Predicate filter — feature_b >= 50")
    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        filters=pc.field("feature_b") >= 50,
    )
    rows = collect(loader)
    feature_b = rows.get("feature_b", [])
    ok_values = check("all returned rows satisfy filter", all(v >= 50 for v in feature_b))
    # Each group of 100 rows has 50 values with feature_b >= 50
    expected = total_rows // 2
    ok_count = check("filtered row count correct",
                     len(feature_b) == expected,
                     f"got {len(feature_b)}, expected {expected}")
    return ok_values and ok_count


def run_output_formats(catalog_config: dict) -> bool:
    print("\n[4] Output formats — numpy, arrow, dict")
    import numpy as np

    ok = True
    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="numpy",
    )
    batch = next(iter(loader))
    ok &= check("numpy: row_id is ndarray", isinstance(batch["row_id"], np.ndarray))

    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="arrow",
        collate_fn=lambda x: x,
    )
    batch = next(iter(loader))
    ok &= check("arrow: batch is RecordBatch", isinstance(batch, pa.RecordBatch))

    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="dict",
        collate_fn=lambda x: x,
    )
    batch = next(iter(loader))
    ok &= check("dict: row_id is list", isinstance(batch["row_id"], list))

    return ok


def run_multi_epoch_shuffle(catalog_config: dict) -> bool:
    print("\n[5] Multi-epoch shuffle — different order each epoch")
    _, dataset = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=1,
        shuffle=True,
        shuffle_seed=99,
    )

    orders = []
    for epoch in range(6):
        dataset.set_epoch(epoch)
        orders.append([sp.file.path for sp in dataset._splits[0].splits])

    distinct = len(set(tuple(o) for o in orders))
    ok = check("shuffle produces varied orders across epochs",
               distinct > 1,
               f"{distinct} distinct orders across 6 epochs")
    return ok


def run_multi_worker(catalog_config: dict, num_workers: int, total_rows: int) -> bool:
    print(f"\n[6] Multi-worker ({num_workers} workers) — disjoint, complete coverage")
    _, dataset = IcebergDataset.create_dataloader(
        table="e2e.test_table",
        catalog_config=catalog_config,
        num_workers=num_workers,
    )

    all_row_ids = []
    for wid in range(num_workers):
        mock_info = MagicMock()
        mock_info.id = wid
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_row_ids.extend(batch["row_id"].tolist())

    ok_count = check("total rows correct",
                     len(all_row_ids) == total_rows,
                     f"got {len(all_row_ids)}, expected {total_rows}")
    ok_dedup = check("no duplicates", len(all_row_ids) == len(set(all_row_ids)))
    return ok_count and ok_dedup


def run_snapshot_time_travel(catalog: SqlCatalog, catalog_config: dict) -> bool:
    print("\n[7] Snapshot time travel — old snapshot returns fewer rows")
    schema = Schema(NestedField(1, "row_id", IntegerType(), required=False))
    snap_table = catalog.create_table("e2e.snap_table", schema=schema)

    snap_table.append(pa.table({"row_id": pa.array(list(range(100)), type=pa.int32())}))
    snap1_id = snap_table.current_snapshot().snapshot_id

    snap_table.append(pa.table({"row_id": pa.array(list(range(100, 200)), type=pa.int32())}))

    loader, _ = IcebergDataset.create_dataloader(
        table="e2e.snap_table",
        catalog_config=catalog_config,
        num_workers=0,
        snapshot_id=snap1_id,
    )
    rows = collect(loader)
    ok = check("snapshot 1 returns only first 100 rows",
               len(rows["row_id"]) == 100 and all(v < 100 for v in rows["row_id"]))
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-files",   type=int, default=4,    help="Iceberg data files")
    parser.add_argument("--rows",        type=int, default=1000, help="Rows per file")
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shuffle",     action="store_true", default=True)
    parser.add_argument("--no-shuffle",  dest="shuffle", action="store_false")
    args = parser.parse_args()

    total_rows = args.num_files * args.rows

    print(f"\n{'='*60}")
    print("  Iceberg E2E Pipeline Test")
    print(f"  files      : {args.num_files}")
    print(f"  rows/file  : {args.rows:,}")
    print(f"  total rows : {total_rows:,}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  shuffle    : {args.shuffle}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        warehouse = os.path.join(tmpdir, "warehouse")
        db_path = os.path.join(tmpdir, "catalog.db")
        os.makedirs(warehouse)

        print("\nSetting up catalog and table...")
        catalog, catalog_config = setup_catalog(warehouse, db_path)
        create_table(catalog, args.num_files, args.rows)
        print(f"  Created table e2e.test_table with {total_rows:,} rows across {args.num_files} files")

        results = []
        results.append(run_basic_read(catalog_config, total_rows, args.batch_size))
        results.append(run_column_projection(catalog_config))
        results.append(run_predicate_filter(catalog_config, total_rows))
        results.append(run_output_formats(catalog_config))
        results.append(run_multi_epoch_shuffle(catalog_config))
        results.append(run_multi_worker(catalog_config, args.num_workers, total_rows))
        results.append(run_snapshot_time_travel(catalog, catalog_config))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    all_ok = all(results)
    print(f"  Results: {passed}/{total} scenarios passed")
    print(f"  Overall: {'✓ ALL PASSED' if all_ok else '✗ FAILURES DETECTED'}")
    print(f"{'='*60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
