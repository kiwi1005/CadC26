from __future__ import annotations

from puzzleplace.actions.executor import ExecutionState, replay_actions
from puzzleplace.data import FloorSetCase

from .pseudo import PseudoTrace


def replay_trace(case: FloorSetCase, trace: PseudoTrace) -> ExecutionState:
    return replay_actions(case, trace.actions)


def compare_positions(case: FloorSetCase, state: ExecutionState, *, tolerance: float = 1e-6) -> dict[str, float | bool]:
    if case.target_positions is None:
        raise ValueError("case.target_positions required for pseudo replay comparison")
    matched = 0
    for block_index in range(case.block_count):
        if block_index not in state.placements:
            continue
        expected = case.target_positions[block_index].tolist()
        actual = list(state.placements[block_index])
        if all(abs(float(a) - float(b)) <= tolerance for a, b in zip(actual, expected, strict=True)):
            matched += 1
    return {
        "matched_blocks": matched,
        "block_count": case.block_count,
        "reconstruction_rate": matched / max(case.block_count, 1),
        "all_blocks_present": len(state.placements) == case.block_count,
    }
