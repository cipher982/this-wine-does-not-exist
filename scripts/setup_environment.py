#!/usr/bin/env python3
"""Bootstrap a development environment for the Wine AI dataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def main() -> None:
    if not PYPROJECT.exists():
        print("pyproject.toml not found; run from repository root.", file=sys.stderr)
        sys.exit(1)

    print("Installing project in editable mode via uv...")
    subprocess.run(["uv", "pip", "install", "-e", ".[dev]"], check=True)
    print("Environment ready. Use `uv run` to execute project commands.")


if __name__ == "__main__":
    main()
