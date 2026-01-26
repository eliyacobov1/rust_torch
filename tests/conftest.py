from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_python_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    if python_dir.exists():
        sys.path.insert(0, str(python_dir))


_ensure_repo_python_on_path()
