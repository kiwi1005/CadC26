from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from puzzleplace.actions.executor import ExecutionState
from puzzleplace.actions.masks import filter_legal_actions
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase

if TYPE_CHECKING:
    from puzzleplace.trajectory.pseudo import PseudoTrace


@dataclass(slots=True)
class CoverageReport:
    total_steps: int
    heuristic_hits: int
    augmented_hits: int

    @property
    def heuristic_coverage(self) -> float:
        return self.heuristic_hits / max(self.total_steps, 1)

    @property
    def augmented_coverage(self) -> float:
        return self.augmented_hits / max(self.total_steps, 1)


def _default_dims(case: FloorSetCase, block_index: int) -> tuple[float, float]:
    area = float(case.area_targets[block_index].item())
    if case.target_positions is not None:
        _x, _y, w, h = [float(v) for v in case.target_positions[block_index].tolist()]
        return w, h
    side = math.sqrt(max(area, 1e-6))
    return side, side


def generate_candidate_actions(
    case: FloorSetCase,
    state: ExecutionState,
    *,
    remaining_blocks: list[int] | None = None,
    teacher_action: TypedAction | None = None,
    include_teacher_hint: bool = False,
) -> list[TypedAction]:
    remaining_blocks = remaining_blocks or [idx for idx in range(case.block_count) if idx not in state.placements]
    candidates: list[TypedAction] = []
    placed_indices = list(state.placements.keys())

    for block_index in remaining_blocks:
        w, h = _default_dims(case, block_index)
        candidates.append(TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=block_index, x=0.0, y=0.0, w=w, h=h, metadata={"source": "heuristic_origin"}))

        boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
        if boundary_code != 0:
            candidates.append(TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=block_index, x=0.0, y=0.0, w=w, h=h, metadata={"source": "boundary_seed"}))

        for target_index in placed_indices:
            candidates.append(TypedAction(ActionPrimitive.PLACE_RELATIVE, block_index=block_index, target_index=target_index, dx=0.0, dy=0.0, w=w, h=h, metadata={"source": "adjacent_right"}))
            candidates.append(TypedAction(ActionPrimitive.PLACE_RELATIVE, block_index=block_index, target_index=target_index, dx=0.0, dy=h, w=w, h=h, metadata={"source": "offset_vertical"}))

    for block_index in placed_indices:
        candidates.append(TypedAction(ActionPrimitive.FREEZE, block_index=block_index, metadata={"source": "freeze_after_place"}))

    if include_teacher_hint and teacher_action is not None:
        hinted = TypedAction(
            primitive=teacher_action.primitive,
            block_index=teacher_action.block_index,
            target_index=teacher_action.target_index,
            boundary_code=teacher_action.boundary_code,
            x=teacher_action.x,
            y=teacher_action.y,
            w=teacher_action.w,
            h=teacher_action.h,
            dx=teacher_action.dx,
            dy=teacher_action.dy,
            metadata={**teacher_action.metadata, "source": "offline_teacher_hint"},
        )
        candidates.append(hinted)

    return filter_legal_actions(case, state, candidates)


def _same_action(a: TypedAction, b: TypedAction) -> bool:
    return (
        a.primitive == b.primitive
        and a.block_index == b.block_index
        and a.target_index == b.target_index
        and a.boundary_code == b.boundary_code
        and a.x == b.x
        and a.y == b.y
        and a.w == b.w
        and a.h == b.h
        and a.dx == b.dx
        and a.dy == b.dy
    )


def compute_expert_candidate_coverage(case: FloorSetCase, traces: list[PseudoTrace]) -> CoverageReport:
    from puzzleplace.actions.executor import replay_actions

    total_steps = 0
    heuristic_hits = 0
    augmented_hits = 0

    for trace in traces:
        state = ExecutionState()
        for action in trace.actions:
            total_steps += 1
            remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
            heuristic = generate_candidate_actions(case, state, remaining_blocks=remaining)
            augmented = generate_candidate_actions(case, state, remaining_blocks=remaining, teacher_action=action, include_teacher_hint=True)
            if any(_same_action(candidate, action) for candidate in heuristic):
                heuristic_hits += 1
            if any(_same_action(candidate, action) for candidate in augmented):
                augmented_hits += 1
            replay_actions(case, [action], initial_state=state)

    return CoverageReport(total_steps=total_steps, heuristic_hits=heuristic_hits, augmented_hits=augmented_hits)
