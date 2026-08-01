"""Strict learned optimizer adapter for effect audits.

Unlike the contest entrypoint, this adapter reports checkpoint failure to the
official evaluator instead of silently attributing the analytic fallback to a
learned lane.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import submission.optimizer as submission_optimizer  # noqa: E402

from hcfp.learned import solve as solve_learned  # noqa: E402
from hcfp.runtime import SolveCase  # noqa: E402


class Optimizer(submission_optimizer.HCFPOptimizer):
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
        checkpoint = os.environ.get("HCFP_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("HCFP_CHECKPOINT is required for learned audits")
        case = SolveCase(
            block_count=int(block_count),
            area_targets=area_targets,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
            constraints=constraints,
            target_positions=target_positions,
        )
        return solve_learned(case, checkpoint=checkpoint, require_checkpoint=True)
