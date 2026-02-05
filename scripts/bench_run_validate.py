#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunSpec:
    run_id: str
    name: str


def write_run(root: Path, spec: RunSpec, schema_version: int) -> None:
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    metadata: Dict[str, object] = {
        "id": spec.run_id,
        "name": spec.name,
        "created_at_unix": now,
        "status": "Completed",
        "status_message": None,
        "config": {},
        "tags": ["bench"],
        "schema_version": schema_version,
    }
    summary: Dict[str, object] = {
        "run_id": spec.run_id,
        "name": spec.name,
        "created_at_unix": now,
        "completed_at_unix": now,
        "status": "Completed",
        "status_message": None,
        "duration_secs": 0.01,
        "metrics": {
            "total_records": 0,
            "first_step": None,
            "last_step": None,
            "first_timestamp_unix": None,
            "last_timestamp_unix": None,
            "metrics": {},
        },
        "telemetry": None,
        "layout": {
            "validations": 0,
            "failures": 0,
            "overlap_failures": 0,
        },
        "schema_version": schema_version,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


def collect_rss_kb(pid: int) -> Optional[int]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def run_validation(binary: str, runs_dir: Path, workers: int) -> tuple[float, Optional[int]]:
    command = [
        binary,
        "runs-validate",
        "--runs-dir",
        str(runs_dir),
        "--workers",
        str(workers),
        "--no-telemetry",
        "--no-metrics",
        "--no-orphaned",
    ]
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    max_rss = 0
    while process.poll() is None:
        rss = collect_rss_kb(process.pid)
        if rss is not None:
            max_rss = max(max_rss, rss)
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    duration = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"validation failed (code={process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return duration, max_rss or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark run governance validation")
    parser.add_argument("--runs", type=int, default=250, help="Number of runs to generate")
    parser.add_argument("--workers", type=int, default=4, help="Validation worker count")
    parser.add_argument(
        "--binary",
        type=str,
        default=str(Path("target/release/rustorch_cli")),
        help="Path to rustorch_cli binary",
    )
    args = parser.parse_args()

    runs_dir = Path("bench_runs")
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.is_dir():
                for item in entry.iterdir():
                    item.unlink()
                entry.rmdir()
        runs_dir.rmdir()
    runs_dir.mkdir(parents=True, exist_ok=True)

    schema_version = int(os.environ.get("RUSTORCH_SCHEMA_VERSION", "2"))
    specs: List[RunSpec] = [
        RunSpec(run_id=f"bench-{idx}", name=f"bench-run-{idx}")
        for idx in range(args.runs)
    ]

    write_start = time.perf_counter()
    for spec in specs:
        write_run(runs_dir, spec, schema_version)
    write_time = time.perf_counter() - write_start

    if not Path(args.binary).exists():
        raise FileNotFoundError(
            f"rustorch_cli not found at {args.binary}. Build with cargo build --release."
        )

    duration, max_rss = run_validation(args.binary, runs_dir, args.workers)
    print("Run Governance Benchmark")
    print(f"runs: {args.runs}")
    print(f"write_time_s: {write_time:.4f}")
    print(f"validate_time_s: {duration:.4f}")
    if max_rss is not None:
        print(f"max_rss_kb: {max_rss}")


if __name__ == "__main__":
    main()
