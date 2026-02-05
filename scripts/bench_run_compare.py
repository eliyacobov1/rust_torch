#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import tempfile
import time
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark run comparison (time + memory delta)."
    )
    parser.add_argument(
        "--binary",
        default="rustorch_cli",
        help="Path to rustorch_cli binary (default: rustorch_cli)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Existing runs directory (default: temp dir populated by this script)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="Number of synthetic runs to generate if runs-dir is empty",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="json",
        help="Output format for comparison command",
    )
    return parser.parse_args()


def collect_run_ids(runs_dir: str) -> List[str]:
    run_ids: List[str] = []
    for entry in os.listdir(runs_dir):
        path = os.path.join(runs_dir, entry)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "run_summary.json")):
            run_ids.append(entry)
    return sorted(run_ids)


def generate_runs(binary: str, runs_dir: str, count: int) -> None:
    for idx in range(count):
        subprocess.run(
            [
                binary,
                "train-linear",
                "--runs-dir",
                runs_dir,
                "--run-name",
                f"bench_run_{idx}",
                "--epochs",
                "4",
                "--samples",
                "128",
                "--features",
                "6",
                "--batch-size",
                "16",
            ],
            check=True,
        )


def run_compare(binary: str, runs_dir: str, run_ids: List[str], output_format: str) -> None:
    run_ids_csv = ",".join(run_ids)
    subprocess.run(
        [
            binary,
            "runs-compare",
            "--runs-dir",
            runs_dir,
            "--run-ids",
            run_ids_csv,
            "--format",
            output_format,
            "--metric-agg",
            "last",
            "--top-k",
            "5",
            "--no-graph",
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir
    temp_dir = None
    if runs_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="rustorch_bench_")
        runs_dir = temp_dir.name

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)

    run_ids = collect_run_ids(runs_dir)
    if not run_ids:
        generate_runs(args.binary, runs_dir, args.count)
        run_ids = collect_run_ids(runs_dir)

    if len(run_ids) < 2:
        print("Need at least two runs to compare.", file=sys.stderr)
        return 1

    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    run_compare(args.binary, runs_dir, run_ids, args.format)
    elapsed = time.perf_counter() - start
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    delta_kb = max(after_usage - before_usage, 0)

    print("\n=== run comparison benchmark ===")
    print(f"runs_dir: {runs_dir}")
    print(f"run_count: {len(run_ids)}")
    print(f"elapsed_seconds: {elapsed:.4f}")
    print(f"memory_delta_kb: {delta_kb}")

    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
