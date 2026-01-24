import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import rust_backend.backend as _  # register backend
import rustorch

def toy(x, y):
    return (x @ y).relu()

if __name__ == "__main__":
    compiled_toy = torch.compile(toy, backend="rust_backend")
    x = torch.randn(16, 32)
    y = torch.randn(32, 8)
    eager_out = toy(x, y)
    compiled_out = compiled_toy(x, y)
    torch.testing.assert_close(compiled_out, eager_out)
    print("OK, eager out:", tuple(eager_out.shape))
    print("OK, compiled out:", tuple(compiled_out.shape))
