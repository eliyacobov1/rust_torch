#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional


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


def run_benchmark(binary: str, stages: int, lanes: int, seed: int) -> tuple[float, Optional[int], Optional[int]]:
    command = [
        binary,
        "--stages",
        str(stages),
        "--lanes",
        str(lanes),
        "--seed",
        str(seed),
    ]
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    max_rss = 0
    start_rss = collect_rss_kb(process.pid)
    while process.poll() is None:
        rss = collect_rss_kb(process.pid)
        if rss is not None:
            max_rss = max(max_rss, rss)
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    duration = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark failed (code={process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    print(stdout.strip())
    max_rss = max_rss or None
    rss_delta = None
    if start_rss is not None and max_rss is not None:
        rss_delta = max_rss - start_rss
    return duration, max_rss, rss_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark plan digest computation")
    parser.add_argument("--stages", type=int, default=10_000, help="Number of stages to generate")
    parser.add_argument("--lanes", type=int, default=8, help="Max lanes for execution planning")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--binary",
        type=str,
        default=str(Path("target/release/plan_digest_perf")),
        help="Path to plan_digest_perf binary",
    )
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(
            f"plan_digest_perf not found at {binary}. Build with cargo build --release --bin plan_digest_perf."
        )

    duration, max_rss, rss_delta = run_benchmark(
        str(binary), args.stages, args.lanes, args.seed
    )
    print("Plan Digest Perf Summary")
    print(f"stages: {args.stages}")
    print(f"lanes: {args.lanes}")
    print(f"duration_s: {duration:.4f}")
    if max_rss is not None:
        print(f"max_rss_kb: {max_rss}")
    if rss_delta is not None:
        print(f"rss_delta_kb: {rss_delta}")


if __name__ == "__main__":
    main()
