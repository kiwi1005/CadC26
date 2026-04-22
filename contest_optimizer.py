#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CONTEST_ROOT = ROOT / "external" / "FloorSet" / "iccad2026contest"

for path in (ROOT, SRC, CONTEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.optimizer.contest import ContestOptimizer

__all__ = ["ContestOptimizer"]
