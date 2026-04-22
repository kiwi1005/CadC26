from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from puzzleplace.data import BOUNDARY_CODES, FloorSetCase

from .schema import ActionPrimitive, TypedAction

Placement = tuple[float, float, float, float]


@dataclass(slots=True)
class ExecutionState:
    placements: dict[int, Placement] = field(default_factory=dict)
    frozen_blocks: set[int] = field(default_factory=set)
    proposed_positions: dict[int, Placement] = field(default_factory=dict)
    shape_assigned: set[int] = field(default_factory=set)
    semantic_placed: set[int] = field(default_factory=set)
    physically_placed: set[int] = field(default_factory=set)
    step: int = 0
    history: list[TypedAction] = field(default_factory=list)
    last_rollout_mode: Literal["semantic", "relaxed", "strict"] = "strict"

    def require_placed(self, block_index: int) -> Placement:
        if block_index not in self.placements:
            raise KeyError(f"Block {block_index} is not placed")
        return self.placements[block_index]

    def require_mutable(self, block_index: int) -> None:
        if block_index in self.frozen_blocks:
            raise ValueError(f"Block {block_index} is frozen")


class ActionExecutor:
    def __init__(self, case: FloorSetCase):
        self.case = case

    def _record_box(self, state: ExecutionState, action: TypedAction, box: Placement) -> None:
        state.placements[action.block_index] = box
        state.proposed_positions[action.block_index] = box
        state.shape_assigned.add(action.block_index)
        state.semantic_placed.add(action.block_index)
        state.physically_placed.add(action.block_index)

    def apply(self, state: ExecutionState, action: TypedAction) -> ExecutionState:
        state.step += 1
        state.history.append(action)
        primitive = action.primitive
        if primitive is ActionPrimitive.PLACE_ABSOLUTE:
            action.require("x", "y", "w", "h")
            assert action.x is not None
            assert action.y is not None
            assert action.w is not None
            assert action.h is not None
            self._record_box(
                state,
                action,
                (float(action.x), float(action.y), float(action.w), float(action.h)),
            )
            return state

        if primitive is ActionPrimitive.MOVE:
            action.require("dx", "dy")
            assert action.dx is not None
            assert action.dy is not None
            state.require_mutable(action.block_index)
            x, y, w, h = state.require_placed(action.block_index)
            self._record_box(
                state,
                action,
                (x + float(action.dx), y + float(action.dy), w, h),
            )
            return state

        if primitive is ActionPrimitive.RESIZE:
            action.require("w", "h")
            assert action.w is not None
            assert action.h is not None
            state.require_mutable(action.block_index)
            x, y, _, _ = state.require_placed(action.block_index)
            self._record_box(
                state,
                action,
                (x, y, float(action.w), float(action.h)),
            )
            return state

        if primitive is ActionPrimitive.PLACE_RELATIVE:
            action.require("target_index", "dx", "dy", "w", "h")
            assert action.target_index is not None
            assert action.dx is not None
            assert action.dy is not None
            assert action.w is not None
            assert action.h is not None
            target = state.require_placed(int(action.target_index))
            tx, ty, tw, th = target
            self._record_box(
                state,
                action,
                (
                    tx + tw + float(action.dx),
                    ty + float(action.dy),
                    float(action.w),
                    float(action.h),
                ),
            )
            return state

        if primitive is ActionPrimitive.ALIGN_BOUNDARY:
            action.require("boundary_code")
            assert action.boundary_code is not None
            state.require_mutable(action.block_index)
            x, y, w, h = state.require_placed(action.block_index)
            code = int(action.boundary_code)
            if code == BOUNDARY_CODES["LEFT"]:
                x = 0.0
            elif code == BOUNDARY_CODES["BOTTOM"]:
                y = 0.0
            elif code == BOUNDARY_CODES["RIGHT"]:
                widths = [px + pw for px, _, pw, _ in state.placements.values()]
                if widths:
                    x = max(widths) - w
            elif code == BOUNDARY_CODES["TOP"]:
                heights = [py + ph for _, py, _, ph in state.placements.values()]
                if heights:
                    y = max(heights) - h
            self._record_box(state, action, (x, y, w, h))
            return state

        if primitive is ActionPrimitive.FREEZE:
            state.frozen_blocks.add(action.block_index)
            return state

        raise NotImplementedError(f"Unsupported action primitive: {primitive}")


def replay_actions(
    case: FloorSetCase,
    actions: list[TypedAction],
    *,
    initial_state: ExecutionState | None = None,
) -> ExecutionState:
    state = initial_state or ExecutionState()
    executor = ActionExecutor(case)
    for action in actions:
        executor.apply(state, action)
    return state
