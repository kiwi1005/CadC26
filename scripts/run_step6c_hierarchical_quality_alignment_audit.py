#!/usr/bin/env python3
"""Shared Step6C quality-after-action helper.

Several Step6D/E/F research diagnostics compare candidate actions by the
official repaired quality after applying exactly one typed action.  The original
quality-alignment audit was split out during the Step6 research iterations; keep
this small compatibility module so later diagnostics can import the shared
helper without reviving the removed audit runner.
"""

from __future__ import annotations

from typing import Any

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.eval import evaluate_positions
from puzzleplace.repair.finalizer import finalize_layout


def _clone_execution_state(state: ExecutionState) -> ExecutionState:
    clone = ExecutionState()
    clone.placements = dict(state.placements)
    clone.proposed_positions = dict(state.proposed_positions)
    clone.semantic_placed = set(state.semantic_placed)
    clone.history = list(state.history)
    return clone


def _quality_from_positions(
    case,
    positions: dict[int, tuple[float, float, float, float]],
) -> dict[str, Any]:
    repair = finalize_layout(case, positions)
    evaluation = evaluate_positions(case, repair.positions, runtime=1.0, median_runtime=1.0)
    return evaluation["quality"]


def _quality_after_action(case, state: ExecutionState, action) -> dict[str, Any]:
    trial = _clone_execution_state(state)
    ActionExecutor(case).apply(trial, action)
    quality = _quality_from_positions(case, trial.proposed_positions)
    return {
        "quality": quality,
        "proposed_positions": dict(trial.proposed_positions),
        "semantic_placed_fraction": len(trial.semantic_placed) / max(case.block_count, 1),
    }
