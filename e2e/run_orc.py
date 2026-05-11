"""
E2E test script for ORC format: StructuredDataset → DataLoader → correctness checks.

Generates ORC test data inline (no external gen_data.py dependency) and validates:
  1. Basic read — all rows returned, no duplicates, no gaps
  2. Column projection
  3. Predicate filter (label == 1, keeps half)
  4. Sub-file splitting — split_rows=100 produces multiple chunks per file
  5. Rank-aware sharding — 3 ranks × 1 worker each, disjoint complete coverage
  6. Multi-worker — 3 workers, disjoint complete coverage

Data schema: row_id (int32), feature_a (float32), feature_b (int32, 0-99), label (int32, 0|1)
ORC files are written with a small stripe_size to force multiple stripes per file.

Usage:
    uv run python e2e/run_orc.py
    uv run python e2e/run_orc.py --num-workers 3 --batch-size 256
"""

import argparse
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.orc as orc

import torch

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

# ---------------------------------------------------------------------------
# File row counts — deliberately unequal; total = 7000 rows
# ---------------------------------------------------------------------------

_FILE_ROW_COUNTS = [500, 1000, 2000, 3000, 500]
_TOTAL_ROWS = sum(_FILE_ROW_COUNTS)  # 7000

# Small stripe size to force multiple stripes per file (8 KiB)
_STRIPE_SIZE = 8 * 1024


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_table(n_rows: int, offset: int) -> pa.Table:
    row_ids = list(range(offset, offset + n_rows))
    return pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "feature_a": pa.array(
                [float(i % 10) / 10.0 for i in row_ids], type=pa.float32()
            ),
            "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
            "label": pa.array([i % 2 for i in row_ids], type=pa.int32()),
        }
    )


def generate_orc_files(out_dir: str) -> list[str]:
    """Write ORC files with unequal row counts and multiple stripes each."""
    paths = []
    offset = 0
    for idx, n_rows in enumerate(_FILE_ROW_COUNTS):
        path = os.path.join(out_dir, f"part_{idx:04d}.orc")
        table = _make_table(n_rows, offset)
        orc.write_table(table, path, stripe_size=_STRIPE_SIZE)
        orf = orc.ORCFile(path)
        nstripes = orf.nstripes
        print(
            f"  wrote {path}  rows={n_rows:,}  stripes={nstripes}"
            f"  size={os.path.getsize(path):,} bytes"
        )
        if nstripes <= 1:
            print(
                f"  WARNING: expected >1 stripes for {os.path.basename(path)}"
                f" but got {nstripes}. Stripe-split test may be less meaningful."
            )
        paths.append(path)
        offset += n_rows
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✓" if condition else "✗ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def collect(loader_or_dataset) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in loader_or_dataset:
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
# Scenario 1 — Basic read
# ---------------------------------------------------------------------------


def run_basic_read(data_dir: str, batch_size: int) -> bool:
    print("\n[1] Basic read — all rows returned, no duplicates, no gaps")
    loader, _ = StructuredDataset.create_dataloader(
        path=data_dir,
        format="orc",
        num_workers=0,
        batch_size=batch_size,
    )
    rows = collect(loader)
    row_ids = sorted(rows.get("row_id", []))
    ok_count = check(
        "row count",
        len(row_ids) == _TOTAL_ROWS,
        f"got {len(row_ids)}, expected {_TOTAL_ROWS}",
    )
    ok_dedup = check("no duplicates", len(row_ids) == len(set(row_ids)))
    ok_range = check("no gaps", row_ids == list(range(_TOTAL_ROWS)))
    return ok_count and ok_dedup and ok_range


# ---------------------------------------------------------------------------
# Scenario 2 — Column projection
# ---------------------------------------------------------------------------


def run_column_projection(data_dir: str) -> bool:
    print("\n[2] Column projection — only feature_a and label returned")
    loader, _ = StructuredDataset.create_dataloader(
        path=data_dir,
        format="orc",
        num_workers=0,
        columns=["feature_a", "label"],
    )
    batch = next(iter(loader))
    ok_present = check(
        "projected columns present", set(batch.keys()) == {"feature_a", "label"}
    )
    ok_absent = check(
        "extra columns absent",
        "row_id" not in batch and "feature_b" not in batch,
    )
    return ok_present and ok_absent


# ---------------------------------------------------------------------------
# Scenario 3 — Predicate filter
# ---------------------------------------------------------------------------


def run_predicate_filter(data_dir: str) -> bool:
    print("\n[3] Predicate filter — label == 1 (keeps half the rows)")
    loader, _ = StructuredDataset.create_dataloader(
        path=data_dir,
        format="orc",
        num_workers=0,
        filters=pc.field("label") == 1,
    )
    rows = collect(loader)
    labels = rows.get("label", [])
    ok_values = check("all returned rows have label == 1", all(v == 1 for v in labels))
    expected = _TOTAL_ROWS // 2
    ok_count = check(
        "filtered row count correct",
        len(labels) == expected,
        f"got {len(labels)}, expected {expected}",
    )
    return ok_values and ok_count


# ---------------------------------------------------------------------------
# Scenario 4 — Sub-file splitting
# ---------------------------------------------------------------------------


def run_sub_file_splitting(data_dir: str, batch_size: int) -> bool:
    print("\n[4] Sub-file splitting — split_rows=100 → multiple chunks per file")
    strategy = TargetSizeSplitStrategy(target_rows=100)
    loader, dataset = StructuredDataset.create_dataloader(
        path=data_dir,
        format="orc",
        num_workers=0,
        batch_size=batch_size,
        split_strategy=strategy,
    )
    all_splits = [sp for shard in dataset._splits for sp in shard.splits]
    ok_multi = check(
        "more chunks than files",
        len(all_splits) > len(_FILE_ROW_COUNTS),
        f"got {len(all_splits)} chunks for {len(_FILE_ROW_COUNTS)} files",
    )
    rows = collect(loader)
    row_ids = sorted(rows.get("row_id", []))
    ok_count = check(
        "row count unchanged",
        len(row_ids) == _TOTAL_ROWS,
        f"got {len(row_ids)}, expected {_TOTAL_ROWS}",
    )
    ok_dedup = check("no duplicates", len(row_ids) == len(set(row_ids)))
    ok_range = check("no gaps", row_ids == list(range(_TOTAL_ROWS)))
    return ok_multi and ok_count and ok_dedup and ok_range


# ---------------------------------------------------------------------------
# Scenario 5 — Rank-aware sharding
# ---------------------------------------------------------------------------


def run_rank_sharding(data_dir: str, batch_size: int) -> bool:
    print("\n[5] Rank-aware sharding — 3 ranks × 1 worker each, disjoint union = all rows")
    num_ranks = 3
    all_row_ids: list[int] = []

    for rank in range(num_ranks):
        _, dataset = StructuredDataset.create_dataloader(
            path=data_dir,
            format="orc",
            num_workers=1,
            batch_size=batch_size,
            num_ranks=num_ranks,
            rank=rank,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_row_ids.extend(batch["row_id"].tolist())

    ok_count = check(
        "total rows correct",
        len(all_row_ids) == _TOTAL_ROWS,
        f"got {len(all_row_ids)}, expected {_TOTAL_ROWS}",
    )
    ok_dedup = check(
        "no duplicates across ranks", len(all_row_ids) == len(set(all_row_ids))
    )
    ok_range = check(
        "complete coverage", sorted(all_row_ids) == list(range(_TOTAL_ROWS))
    )
    return ok_count and ok_dedup and ok_range


# ---------------------------------------------------------------------------
# Scenario 6 — Multi-worker
# ---------------------------------------------------------------------------


def run_multi_worker(data_dir: str, num_workers: int, batch_size: int) -> bool:
    print(f"\n[6] Multi-worker ({num_workers} workers) — disjoint complete coverage")
    _, dataset = StructuredDataset.create_dataloader(
        path=data_dir,
        format="orc",
        num_workers=num_workers,
        batch_size=batch_size,
    )
    all_row_ids: list[int] = []
    for wid in range(num_workers):
        mock_info = MagicMock()
        mock_info.id = wid
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_row_ids.extend(batch["row_id"].tolist())

    ok_count = check(
        "total rows correct",
        len(all_row_ids) == _TOTAL_ROWS,
        f"got {len(all_row_ids)}, expected {_TOTAL_ROWS}",
    )
    ok_dedup = check(
        "no duplicates across workers", len(all_row_ids) == len(set(all_row_ids))
    )
    ok_range = check(
        "complete coverage", sorted(all_row_ids) == list(range(_TOTAL_ROWS))
    )
    return ok_count and ok_dedup and ok_range


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E test for ORC format with StructuredDataset"
    )
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  ORC E2E Test")
    print(f"  files      : {len(_FILE_ROW_COUNTS)}  ({_FILE_ROW_COUNTS})")
    print(f"  total rows : {_TOTAL_ROWS:,}")
    print(f"  stripe_size: {_STRIPE_SIZE // 1024} KiB  (forces multiple stripes)")
    print(f"  num_workers: {args.num_workers}")
    print(f"  batch_size : {args.batch_size:,}")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nGenerating ORC test data...")
        generate_orc_files(tmpdir)
        print(f"  Total: {_TOTAL_ROWS:,} rows across {len(_FILE_ROW_COUNTS)} files\n")

        results = []
        results.append(run_basic_read(tmpdir, args.batch_size))
        results.append(run_column_projection(tmpdir))
        results.append(run_predicate_filter(tmpdir))
        results.append(run_sub_file_splitting(tmpdir, args.batch_size))
        results.append(run_rank_sharding(tmpdir, args.batch_size))
        results.append(run_multi_worker(tmpdir, args.num_workers, args.batch_size))

    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    all_ok = all(results)
    print(f"  Results: {passed}/{total} scenarios passed")
    print(f"  Overall: {'✓ ALL PASSED' if all_ok else '✗ FAILURES DETECTED'}")
    print(f"{'=' * 60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
