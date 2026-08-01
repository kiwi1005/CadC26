"""Official-compatible fallback-only adapter for P0 validation audits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from hcfp.fallback import safe_fallback  # noqa: E402
from hcfp.runtime import HCFPRuntime  # noqa: E402
from submission.optimizer import HCFPOptimizer  # noqa: E402


class Optimizer(HCFPOptimizer):
    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)
        self.runtime = HCFPRuntime(solver=safe_fallback)
