from __future__ import annotations

from dataclasses import dataclass

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.geometry.boxes import pairwise_intersection_area

from .executor import ExecutionState
from .schema import ActionPrimitive, TypedAction


@dataclass(slots=True)
class MaskDecision:
    allowed: bool
    reason: str


def _candidate_box(state: ExecutionState, action: TypedAction) -> tuple[float, float, float, float] | None:
    if action.primitive is ActionPrimitive.PLACE_ABSOLUTE:
        action.require("x", "y", "w", "h")
        return (float(action.x), float(action.y), float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.PLACE_RELATIVE:
        action.require("target_index", "dx", "dy", "w", "h")
        tx, ty, tw, _th = state.require_placed(int(action.target_index))
        return (tx + tw + float(action.dx), ty + float(action.dy), float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.MOVE:
        action.require("dx", "dy")
        x, y, w, h = state.require_placed(action.block_index)
        return (x + float(action.dx), y + float(action.dy), w, h)
    if action.primitive is ActionPrimitive.RESIZE:
        action.require("w", "h")
        x, y, _w, _h = state.require_placed(action.block_index)
        return (x, y, float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.ALIGN_BOUNDARY:
        return state.require_placed(action.block_index)
    return None


def check_action_mask(case: FloorSetCase, state: ExecutionState, action: TypedAction) -> MaskDecision:
    if not (0 <= action.block_index < case.block_count):
        return MaskDecision(False, "invalid block index")

    fixed = bool(case.constraints[action.block_index, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[action.block_index, ConstraintColumns.PREPLACED].item())
    immutable = fixed or preplaced or action.block_index in state.frozen_blocks

    if action.primitive in {ActionPrimitive.MOVE, ActionPrimitive.RESIZE, ActionPrimitive.ALIGN_BOUNDARY} and immutable:
        return MaskDecision(False, "immutable block cannot be mutated")

    if action.primitive is ActionPrimitive.FREEZE and action.block_index not in state.placements:
        return MaskDecision(False, "cannot freeze unplaced block")

    try:
        box = _candidate_box(state, action)
    except (KeyError, ValueError) as exc:
        return MaskDecision(False, str(exc))

    if box is not None:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return MaskDecision(False, "non-positive dimensions")
        target_area = float(case.area_targets[action.block_index].item())
        if target_area > 0 and abs((w * h) - target_area) / target_area > 0.01:
            return MaskDecision(False, "area tolerance exceeded")
        if (fixed or preplaced) and case.target_positions is not None:
            tx, ty, tw, th = [float(v) for v in case.target_positions[action.block_index].tolist()]
            if abs(w - tw) > 1e-4 or abs(h - th) > 1e-4:
                return MaskDecision(False, "fixed/preplaced dimensions must match target reference")
            if preplaced and (abs(x - tx) > 1e-4 or abs(y - ty) > 1e-4):
                return MaskDecision(False, "preplaced location must match target reference")
        for other_idx, other_box in state.placements.items():
            if other_idx == action.block_index:
                continue
            if pairwise_intersection_area(
                __import__('torch').tensor(box, dtype=__import__('torch').float32),
                __import__('torch').tensor(other_box, dtype=__import__('torch').float32),
            ) > 1e-9:
                return MaskDecision(False, f"overlaps block {other_idx}")

    return MaskDecision(True, "legal")


def filter_legal_actions(case: FloorSetCase, state: ExecutionState, actions: list[TypedAction]) -> list[TypedAction]:
    return [action for action in actions if check_action_mask(case, state, action).allowed]
