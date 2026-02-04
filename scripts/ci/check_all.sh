#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found in PATH" >&2
    exit 1
  fi
}

run_rust=1
run_python=1
run_bench=1

for arg in "$@"; do
  case "$arg" in
    --skip-rust) run_rust=0 ;;
    --skip-python) run_python=0 ;;
    --skip-bench) run_bench=0 ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/ci/check_all.sh [--skip-rust] [--skip-python] [--skip-bench]

Runs a local preflight equivalent of CI jobs:
  - Rust tests (no default features)
  - Python build/tests/backend smoke test
  - Benchmark regression script
EOF
      exit 0
      ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      exit 1
      ;;
  esac
done

if [[ "$run_rust" -eq 1 ]]; then
  echo "==> Rust tests (no Python bindings)"
  require_cmd cargo
  cargo test --no-default-features
fi

if [[ "$run_python" -eq 1 ]]; then
  echo "==> Python build + tests"
  require_cmd python
  require_cmd cargo

  python -m pip install -U pip wheel setuptools
  python -m pip install maturin numpy pytest
  python -m pip install torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

  maturin build --release --out dist
  python -m pip install --force-reinstall dist/*.whl

  PYTHONPATH=python pytest -q tests/test_tensor.py tests/test_fx_runner.py
  PYTHONPATH=python python python/rust_backend/backend_check.py
fi

if [[ "$run_bench" -eq 1 ]]; then
  echo "==> Benchmarks"
  require_cmd cargo
  scripts/ci/run_benchmarks.sh
fi

echo "==> CI preflight completed successfully"
