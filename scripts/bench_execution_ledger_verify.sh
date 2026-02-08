#!/usr/bin/env bash
set -euo pipefail

export EXECUTION_LEDGER_PERF_STAGES=${EXECUTION_LEDGER_PERF_STAGES:-2000}
export EXECUTION_LEDGER_PERF_LANES=${EXECUTION_LEDGER_PERF_LANES:-8}

cargo run --release --bin execution_ledger_verify_perf
