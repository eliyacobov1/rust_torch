#!/usr/bin/env bash
set -euo pipefail

NODES=${GOVERNANCE_PERF_NODES:-2000}
SEED=${GOVERNANCE_PERF_SEED:-2025}

GOVERNANCE_PERF_NODES="$NODES" \
GOVERNANCE_PERF_SEED="$SEED" \
  cargo run --release --bin governance_perf
