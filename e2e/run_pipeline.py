"""
E2E pipeline test: StructuredDataset → DataLoader → transformer embedding → tqdm progress.

Runs the full real DataLoader pipeline with real worker subprocesses.
Applies a small transformer encoder to each batch to simulate realistic GPU compute,
keeping the pipeline busy long enough to observe throughput and worker utilisation.

Usage:
    # Generate data first
    uv run python e2e/gen_data.py --out-dir /tmp/e2e_data --files 20 --rows 100000

    # Run e2e test
    uv run python e2e/run_pipeline.py --data-dir /tmp/e2e_data --num-workers 4
    uv run python e2e/run_pipeline.py --data-dir /tmp/e2e_data --num-workers 4 --unequal
    uv run python e2e/run_pipeline.py --data-dir /tmp/e2e_data --num-workers 0  # single process
"""
import argparse
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy


# ---------------------------------------------------------------------------
# Tiny transformer encoder — runs on CPU, adds realistic compute per batch
# ---------------------------------------------------------------------------

class EmbeddingModel(nn.Module):
    """64-feature input → 32-dim embedding via a single transformer encoder layer."""

    def __init__(self, in_features: int = 64, embed_dim: int = 32, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=64,
            batch_first=True,
        )
        self.head = nn.Linear(embed_dim, 10)  # 10-class classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64)
        x = self.proj(x)           # (batch, embed_dim)
        x = x.unsqueeze(1)         # (batch, 1, embed_dim) — seq_len=1
        x = self.encoder(x)        # (batch, 1, embed_dim)
        x = x.squeeze(1)           # (batch, embed_dim)
        return self.head(x)        # (batch, 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_features(batch: dict) -> torch.Tensor:
    """Stack feat_00..feat_63 into a single (batch, 64) tensor."""
    cols = [batch[f"feat_{i:02d}"] for i in range(64)]
    return torch.stack(cols, dim=1).float()


def format_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp/e2e_data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--target-mb", type=int, default=32, help="Target split size in MiB")
    parser.add_argument("--target-rows", type=int, default=None, help="Target rows per chunk (overrides --target-mb)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*60}")
    print(f"  E2E Pipeline Test")
    print(f"  data_dir    : {args.data_dir}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  batch_size  : {args.batch_size:,}")
    print(f"  target_mb   : {args.target_mb} MiB")
    print(f"  target_rows : {args.target_rows if args.target_rows else 'not set'}")
    print(f"  epochs      : {args.epochs}")
    print(f"  shuffle     : {args.shuffle}")
    print(f"{'='*60}\n")

    model = EmbeddingModel()
    model.eval()

    strategy = TargetSizeSplitStrategy(
        target_bytes=args.target_mb * 1024 * 1024,
        target_rows=args.target_rows,
        shuffle=args.shuffle,
        seed=42,
    )

    loader, dataset = StructuredDataset.create_dataloader(
        path=args.data_dir,
        format="parquet",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        split_strategy=strategy,
    )

    # Print split plan
    splits = dataset._splits
    print(f"Split plan: {len(splits)} shard(s) across {args.num_workers or 1} worker(s)")
    for s in splits:
        total_rows = sum(
            (sp.row_range.length if sp.row_range else sp.file.record_count or 0)
            for sp in s.splits
        )
        total_bytes = sum(
            sp.file.file_size or 0 for sp in s.splits
        )
        print(f"\n  Shard {s.id}: {len(s.splits)} split(s)  "
              f"~{format_num(total_rows)} rows  ~{total_bytes // 1024}KB")
        for sp in s.splits:
            fname = os.path.basename(sp.file.path)
            if sp.row_range is not None:
                rr = sp.row_range
                print(f"    {fname}  rows [{rr.offset:,} – {rr.offset + rr.length - 1:,}]  "
                      f"({format_num(rr.length)} rows)")
            else:
                rec = sp.file.record_count or 0
                print(f"    {fname}  full file  ({format_num(rec)} rows)")
    print()

    epoch_stats = []

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)

        rows_seen = 0
        batches_seen = 0
        row_ids_seen = []
        t0 = time.perf_counter()

        desc = f"Epoch {epoch+1}/{args.epochs}"
        with tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True) as pbar:
            for batch in pbar:
                features = collect_features(batch)
                labels = batch["label"]
                row_ids = batch["row_id"].tolist()

                with torch.no_grad():
                    logits = model(features)
                    _ = torch.argmax(logits, dim=1)

                rows_seen += len(labels)
                batches_seen += 1
                row_ids_seen.extend(row_ids)

                elapsed = time.perf_counter() - t0
                rows_per_sec = rows_seen / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    rows=format_num(rows_seen),
                    rows_s=f"{format_num(int(rows_per_sec))}/s",
                    batches=batches_seen,
                )

        elapsed = time.perf_counter() - t0

        # Correctness check
        unique_ids = set(row_ids_seen)
        duplicates = len(row_ids_seen) - len(unique_ids)
        gaps = sorted(unique_ids) != list(range(min(unique_ids), max(unique_ids) + 1)) if unique_ids else False

        print(f"\n  Epoch {epoch+1} results:")
        print(f"    rows        : {rows_seen:,}")
        print(f"    batches     : {batches_seen:,}")
        print(f"    elapsed     : {elapsed:.2f}s")
        print(f"    throughput  : {rows_seen/elapsed:,.0f} rows/s")
        print(f"    duplicates  : {duplicates}  {'✗ FAIL' if duplicates else '✓ OK'}")
        print(f"    gaps        : {'YES ✗ FAIL' if gaps else 'none ✓ OK'}")

        epoch_stats.append({
            "epoch": epoch,
            "rows": rows_seen,
            "batches": batches_seen,
            "elapsed": elapsed,
            "duplicates": duplicates,
            "gaps": gaps,
        })

    # Cross-epoch check: same total rows each epoch
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_ok = True
    for s in epoch_stats:
        ok = s["duplicates"] == 0 and not s["gaps"]
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  Epoch {s['epoch']+1}: {s['rows']:,} rows  {s['batches']:,} batches  "
              f"{s['elapsed']:.2f}s  {s['rows']/s['elapsed']:,.0f} rows/s  {status}")

    row_counts = [s["rows"] for s in epoch_stats]
    if len(set(row_counts)) > 1:
        print(f"\n  ✗ FAIL: row counts differ across epochs: {row_counts}")
        all_ok = False
    else:
        print(f"\n  ✓ Row counts consistent across epochs: {row_counts[0]:,}")

    print(f"\n  Overall: {'✓ ALL PASSED' if all_ok else '✗ FAILURES DETECTED'}")
    print(f"{'='*60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
