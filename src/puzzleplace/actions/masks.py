from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.geometry.boxes import pairwise_intersection_area

from .executor import ExecutionState
from .schema import ActionPrimitive, TypedAction

CandidateMode = Literal["semantic", "relaxed", "strict"]


@dataclass(slots=True)
class MaskDecision:
    allowed: bool
    reason: str


def _candidate_box(
    state: ExecutionState, action: TypedAction
) -> tuple[float, float, float, float] | None:
    if action.primitive is ActionPrimitive.PLACE_ABSOLUTE:
        action.require("x", "y", "w", "h")
        assert action.x is not None and action.y is not None
        assert action.w is not None and action.h is not None
        return (float(action.x), float(action.y), float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.PLACE_RELATIVE:
        action.require("target_index", "dx", "dy", "w", "h")
        assert action.target_index is not None
        assert action.dx is not None and action.dy is not None
        assert action.w is not None and action.h is not None
        tx, ty, tw, _th = state.require_placed(int(action.target_index))
        return (tx + tw + float(action.dx), ty + float(action.dy), float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.MOVE:
        action.require("dx", "dy")
        assert action.dx is not None and action.dy is not None
        x, y, w, h = state.require_placed(action.block_index)
        return (x + float(action.dx), y + float(action.dy), w, h)
    if action.primitive is ActionPrimitive.RESIZE:
        action.require("w", "h")
        assert action.w is not None and action.h is not None
        x, y, _w, _h = state.require_placed(action.block_index)
        return (x, y, float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.ALIGN_BOUNDARY:
        return state.require_placed(action.block_index)
    return None


def estimate_action_violations(
    case: FloorSetCase,
    state: ExecutionState,
    action: TypedAction,
) -> dict[str, float]:
    try:
        box = _candidate_box(state, action)
    except (KeyError, ValueError):
        return {
            "overlap_pairs": 0.0,
            "total_overlap_area": 0.0,
            "area_error": 0.0,
            "boundary_distance": 0.0,
            "connectivity_proxy_cost": 0.0,
        }

    if box is None:
        return {
            "overlap_pairs": 0.0,
            "total_overlap_area": 0.0,
            "area_error": 0.0,
            "boundary_distance": 0.0,
            "connectivity_proxy_cost": 0.0,
        }

    target_area = float(case.area_targets[action.block_index].item())
    area_error = 0.0
    if target_area > 0:
        area_error = abs((box[2] * box[3]) - target_area) / target_area
    overlap_pairs = 0.0
    total_overlap_area = 0.0
    for other_idx, other_box in state.placements.items():
        if other_idx == action.block_index:
            continue
        overlap_area = float(
            pairwise_intersection_area(
                __import__("torch").tensor(box, dtype=__import__("torch").float32),
                __import__("torch").tensor(other_box, dtype=__import__("torch").float32),
            )
        )
        if overlap_area > 1e-9:
            overlap_pairs += 1.0
            total_overlap_area += overlap_area

    boundary_distance = 0.0
    code = int(case.constraints[action.block_index, ConstraintColumns.BOUNDARY].item())
    if code == 1:
        boundary_distance = abs(box[0])
    elif code == 8:
        boundary_distance = abs(box[1])
    elif state.placements:
        max_x = max(x + width for x, _y, width, _h in state.placements.values())
        max_y = max(y + height for _x, y, _w, height in state.placements.values())
        if code == 2:
            boundary_distance = abs(max_x - (box[0] + box[2]))
        elif code == 4:
            boundary_distance = abs(max_y - (box[1] + box[3]))

    connectivity_proxy_cost = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        if src == -1 or dst == -1:
            continue
        src_idx = int(src)
        dst_idx = int(dst)
        if src_idx == action.block_index and dst_idx in state.placements:
            ox, oy, ow, oh = state.placements[dst_idx]
            connectivity_proxy_cost += float(weight) * (
                abs((box[0] + box[2] / 2.0) - (ox + ow / 2.0))
                + abs((box[1] + box[3] / 2.0) - (oy + oh / 2.0))
            )
        elif dst_idx == action.block_index and src_idx in state.placements:
            ox, oy, ow, oh = state.placements[src_idx]
            connectivity_proxy_cost += float(weight) * (
                abs((box[0] + box[2] / 2.0) - (ox + ow / 2.0))
                + abs((box[1] + box[3] / 2.0) - (oy + oh / 2.0))
            )
    return {
        "overlap_pairs": overlap_pairs,
        "total_overlap_area": total_overlap_area,
        "area_error": float(area_error),
        "boundary_distance": float(boundary_distance),
        "connectivity_proxy_cost": float(connectivity_proxy_cost),
    }


def check_action_mask(
    case: FloorSetCase,
    state: ExecutionState,
    action: TypedAction,
    *,
    mode: CandidateMode = "strict",
) -> MaskDecision:
    if not (0 <= action.block_index < case.block_count):
        return MaskDecision(False, "invalid block index")

    fixed = bool(case.constraints[action.block_index, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[action.block_index, ConstraintColumns.PREPLACED].item())
    immutable = fixed or preplaced or action.block_index in state.frozen_blocks

    if (
        action.primitive
        in {ActionPrimitive.MOVE, ActionPrimitive.RESIZE, ActionPrimitive.ALIGN_BOUNDARY}
        and immutable
    ):
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
        if mode == "strict" and target_area > 0 and abs((w * h) - target_area) / target_area > 0.01:
            return MaskDecision(False, "area tolerance exceeded")
        if (fixed or preplaced) and case.target_positions is not None:
            tx, ty, tw, th = [float(v) for v in case.target_positions[action.block_index].tolist()]
            if abs(w - tw) > 1e-4 or abs(h - th) > 1e-4:
                return MaskDecision(False, "fixed/preplaced dimensions must match target reference")
            if preplaced and (abs(x - tx) > 1e-4 or abs(y - ty) > 1e-4):
                return MaskDecision(False, "preplaced location must match target reference")
        if mode == "semantic":
            return MaskDecision(True, "semantic_type_legal")
        for other_idx, other_box in state.placements.items():
            if other_idx == action.block_index:
                continue
            overlap_area = pairwise_intersection_area(
                __import__("torch").tensor(box, dtype=__import__("torch").float32),
                __import__("torch").tensor(other_box, dtype=__import__("torch").float32),
            )
            if mode == "strict" and overlap_area > 1e-9:
                return MaskDecision(False, f"overlaps block {other_idx}")

    return MaskDecision(True, "legal")


def filter_legal_actions(
    case: FloorSetCase,
    state: ExecutionState,
    actions: list[TypedAction],
    *,
    mode: CandidateMode = "strict",
) -> list[TypedAction]:
    return [
        action for action in actions if check_action_mask(case, state, action, mode=mode).allowed
    ]
