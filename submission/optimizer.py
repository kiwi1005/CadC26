"""Official submission optimizer entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcfp.runtime import HCFPRuntime  # noqa: E402


class HCFPOptimizer:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.runtime = HCFPRuntime()

    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ):
        return self.runtime.solve(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )


_OPTIMIZER = HCFPOptimizer()


def solve(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
):
    return _OPTIMIZER.solve(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
    )


Optimizer = HCFPOptimizer
