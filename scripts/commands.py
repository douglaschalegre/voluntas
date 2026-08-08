"""Development commands exposed through ``uv run``."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, *arguments: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode


def lint() -> int:
    """Run the repository lint checks."""
    return _run_module("ruff", "check", "voluntas", "tests", "scripts")


def tests() -> int:
    """Run the repository test suite."""
    return _run_module("pytest")
