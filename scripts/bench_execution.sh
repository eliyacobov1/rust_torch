#!/usr/bin/env bash
set -euo pipefail

export EXECUTION_PERF_NODES=${EXECUTION_PERF_NODES:-2000}
export EXECUTION_PERF_SEED=${EXECUTION_PERF_SEED:-2025}
export EXECUTION_PERF_LANES=${EXECUTION_PERF_LANES:-8}

cargo run --quiet --bin execution_perf
