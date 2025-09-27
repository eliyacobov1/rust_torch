#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -U pip wheel setuptools maturin numpy
maturin develop --release
python -c "import rustorch; print('rustorch imported OK')"