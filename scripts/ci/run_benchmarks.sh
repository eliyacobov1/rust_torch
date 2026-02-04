#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "==> Running Rust benchmarks (performance regression baseline)"
BENCH_OUTPUT_DIR="$ROOT_DIR/artifacts"
mkdir -p "$BENCH_OUTPUT_DIR"
export RUSTORCH_BENCH_TELEMETRY="$BENCH_OUTPUT_DIR/benchmarks.jsonl"

cargo bench --no-default-features --bench bench_elementwise --bench bench_layout_validation -- \
  --sample-size 10 \
  --warm-up-time 0.1 \
  --measurement-time 0.2

echo "==> Telemetry written to $RUSTORCH_BENCH_TELEMETRY"
