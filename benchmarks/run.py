#!/usr/bin/env python3
"""Benchmark runner — entry point for all scenarios.

Usage:
    # Generate data first
    python benchmarks/gen_data.py --out-dir /tmp/bench_data --dataset small

    # Run all scenarios
    python benchmarks/run.py --data-dir /tmp/bench_data

    # Run specific scenarios
    python benchmarks/run.py --data-dir /tmp/bench_data --scenarios S1 S2 S3

    # CI mode (tiny dataset, fewer runs)
    python benchmarks/run.py --data-dir /tmp/bench_data --ci

    # Save results and compare against baseline
    python benchmarks/run.py --data-dir /tmp/bench_data --output-dir benchmarks/results

    # Promote latest results to baseline
    python benchmarks/run.py --data-dir /tmp/bench_data --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime

from benchmarks._common import verify_manifest
from benchmarks.scenarios import ALL_SCENARIOS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline.json")

CI_DATASETS = {
    "S1": "tiny", "S2": "unequal", "S3": "single_large",
    "S4": "tiny", "S5": "tiny",    "S6": "root",
    "S7": "tiny", "S8": "tiny",
}
DEFAULT_DATASETS = {
    "S1": "large", "S2": "unequal", "S3": "single_large",
    "S4": "large", "S5": "large",   "S6": "root",
    "S7": "small", "S8": "small",
}


def _run_metadata() -> dict:
    import importlib.metadata
    try:
        version = importlib.metadata.version("torch-dataloader-utils")
    except Exception:
        version = "dev"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "library_version": version,
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "python": platform.python_version(),
    }


def _dataset_dir(data_dir: str, scenario_id: str, dataset_name: str, uses_root: bool) -> str:
    if uses_root:
        return data_dir
    return os.path.join(data_dir, dataset_name)


def _check_regressions(current: dict, baseline: dict) -> list[str]:
    failures = []
    threshold = 0.90
    for sid, scenario in current.get("scenarios", {}).items():
        if sid not in baseline.get("scenarios", {}):
            continue
        bl_scenario = baseline["scenarios"][sid]
        for impl_key, impl_data in scenario.items():
            if impl_key in ("description", "dataset", "total_rows", "num_files", "total_bytes"):
                continue
            if not isinstance(impl_data, dict):
                continue
            for cfg_key, cfg in impl_data.items():
                if not isinstance(cfg, dict) or "rows_per_sec" not in cfg:
                    continue
                try:
                    cur_rps = cfg["rows_per_sec"]["median"]
                    bl_rps = bl_scenario[impl_key][cfg_key]["rows_per_sec"]["median"]
                    if cur_rps < bl_rps * threshold:
                        pct = 100 * cur_rps / bl_rps
                        failures.append(
                            f"  {sid}/{impl_key}/{cfg_key}: "
                            f"{cur_rps:,} rows/sec vs baseline {bl_rps:,} ({pct:.0f}%)"
                        )
                except (KeyError, TypeError):
                    continue
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run torch-dataloader-utils benchmarks")
    parser.add_argument("--data-dir", default="/tmp/bench_data", help="Root data directory")
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(ALL_SCENARIOS) + ["all"], default=["all"]
    )
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--ci", action="store_true", help="CI mode: tiny data, fewer runs")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-warmup", type=int, default=None)
    parser.add_argument(
        "--update-baseline", action="store_true", help="Promote results to baseline.json"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip manifest checksum check")
    args = parser.parse_args()

    scenario_ids = list(ALL_SCENARIOS) if "all" in args.scenarios else args.scenarios
    dataset_map = CI_DATASETS if args.ci else DEFAULT_DATASETS
    n_runs = args.n_runs or (3 if args.ci else 5)
    n_warmup = args.n_warmup or (0 if args.ci else 1)

    # Verify manifests for all datasets we'll use
    if not args.skip_verify:
        verified: set[str] = set()
        for sid in scenario_ids:
            module, uses_root = ALL_SCENARIOS[sid]
            ds = dataset_map[sid]
            d = _dataset_dir(args.data_dir, sid, ds, uses_root)
            check_dir = d if uses_root else d
            # For root-dir scenarios, verify the subdatasets that exist
            if uses_root:
                for sub in ["tiny", "small", "medium", "large"]:
                    sub_dir = os.path.join(d, sub)
                    if os.path.isdir(sub_dir) and sub_dir not in verified:
                        print(f"Verifying {sub_dir} ...", end=" ", flush=True)
                        verify_manifest(sub_dir)
                        print("ok")
                        verified.add(sub_dir)
            else:
                if d not in verified:
                    print(f"Verifying {d} ...", end=" ", flush=True)
                    verify_manifest(d)
                    print("ok")
                    verified.add(d)

    # Run scenarios
    all_results: dict = {"meta": _run_metadata(), "scenarios": {}}

    for sid in scenario_ids:
        module, uses_root = ALL_SCENARIOS[sid]
        ds = dataset_map[sid]
        d = _dataset_dir(args.data_dir, sid, ds, uses_root)
        print(f"\n{'='*60}")
        print(f"Running {sid}: {module.__name__.split('.')[-1]}")
        print(f"  data_dir : {d}")
        print(f"  n_runs   : {n_runs}  n_warmup: {n_warmup}")
        print(f"{'='*60}")
        try:
            result = module.run(d, n_warmup=n_warmup, n_runs=n_runs)
            all_results["scenarios"][sid] = result
            total = result.get("total_rows")
            rows_str = f"{total:,}" if isinstance(total, int) else "?"
            print(f"  done — {rows_str} rows")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            all_results["scenarios"][sid] = {"error": str(e)}

    # Write timestamped results
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Regression check
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        failures = _check_regressions(all_results, baseline)
        if failures:
            print("\n⚠️  REGRESSION DETECTED (threshold: 90% of baseline):")
            for f in failures:
                print(f)
            if not args.update_baseline:
                return 1
        else:
            print("\n✓ No regressions vs baseline.")

    if args.update_baseline:
        shutil.copy(out_path, BASELINE_PATH)
        print(f"Baseline updated: {BASELINE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
