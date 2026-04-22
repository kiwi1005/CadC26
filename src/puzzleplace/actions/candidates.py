from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from puzzleplace.actions.executor import ExecutionState
from puzzleplace.actions.masks import (
    CandidateMode,
    estimate_action_violations,
    filter_legal_actions,
)
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase

if TYPE_CHECKING:
    from puzzleplace.trajectory.pseudo import PseudoTrace


@dataclass(slots=True)
class CoverageReport:
    total_steps: int
    semantic_hits: int
    relaxed_hits: int
    strict_hits: int
    teacher_hint_hits: int

    @property
    def semantic_coverage(self) -> float:
        return self.semantic_hits / max(self.total_steps, 1)

    @property
    def relaxed_coverage(self) -> float:
        return self.relaxed_hits / max(self.total_steps, 1)

    @property
    def strict_coverage(self) -> float:
        return self.strict_hits / max(self.total_steps, 1)

    @property
    def teacher_hint_coverage(self) -> float:
        return self.teacher_hint_hits / max(self.total_steps, 1)

    @property
    def heuristic_hits(self) -> int:
        return self.semantic_hits

    @property
    def augmented_hits(self) -> int:
        return self.teacher_hint_hits

    @property
    def heuristic_coverage(self) -> float:
        return self.semantic_coverage

    @property
    def augmented_coverage(self) -> float:
        return self.teacher_hint_coverage


def _bbox(
    placements: dict[int, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not placements:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(x for x, _y, _w, _h in placements.values())
    min_y = min(y for _x, y, _w, _h in placements.values())
    max_x = max(x + width for x, _y, width, _h in placements.values())
    max_y = max(y + height for _x, y, _w, height in placements.values())
    return (min_x, min_y, max_x, max_y)


def _candidate_with_metrics(
    case: FloorSetCase,
    state: ExecutionState,
    action: TypedAction,
    *,
    intent_type: str,
) -> TypedAction:
    metrics = estimate_action_violations(case, state, action)
    action.metadata.update(metrics)
    action.metadata["intent_type"] = intent_type
    return action


def _semantic_candidates_for_block(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
) -> list[TypedAction]:
    width, height = _default_dims(case, block_index)
    candidates: list[TypedAction] = []
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = _bbox(state.placements)

    candidates.append(
        _candidate_with_metrics(
            case,
            state,
            TypedAction(
                ActionPrimitive.PLACE_ABSOLUTE,
                block_index=block_index,
                x=bbox_max_x + 1.0,
                y=bbox_min_y,
                w=width,
                h=height,
                metadata={"source": "semantic_strip_right"},
            ),
            intent_type="place_relative_strip",
        )
    )
    candidates.append(
        _candidate_with_metrics(
            case,
            state,
            TypedAction(
                ActionPrimitive.PLACE_ABSOLUTE,
                block_index=block_index,
                x=bbox_min_x,
                y=bbox_max_y + 1.0,
                w=width,
                h=height,
                metadata={"source": "semantic_strip_top"},
            ),
            intent_type="place_relative_strip",
        )
    )

    boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
    if boundary_code == 1:
        boundary_action = TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=block_index,
            x=0.0,
            y=bbox_min_y,
            w=width,
            h=height,
            metadata={"source": "semantic_boundary_left"},
        )
        candidates.append(
            _candidate_with_metrics(
                case,
                state,
                boundary_action,
                intent_type="snap_to_boundary",
            )
        )
    elif boundary_code == 2:
        boundary_action = TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=block_index,
            x=max(bbox_max_x - width, 0.0),
            y=bbox_min_y,
            w=width,
            h=height,
            metadata={"source": "semantic_boundary_right"},
        )
        candidates.append(
            _candidate_with_metrics(
                case,
                state,
                boundary_action,
                intent_type="snap_to_boundary",
            )
        )
    elif boundary_code == 4:
        boundary_action = TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=block_index,
            x=bbox_min_x,
            y=max(bbox_max_y - height, 0.0),
            w=width,
            h=height,
            metadata={"source": "semantic_boundary_top"},
        )
        candidates.append(
            _candidate_with_metrics(
                case,
                state,
                boundary_action,
                intent_type="snap_to_boundary",
            )
        )
    elif boundary_code == 8:
        boundary_action = TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=block_index,
            x=bbox_min_x,
            y=0.0,
            w=width,
            h=height,
            metadata={"source": "semantic_boundary_bottom"},
        )
        candidates.append(
            _candidate_with_metrics(
                case,
                state,
                boundary_action,
                intent_type="snap_to_boundary",
            )
        )

    if len(case.pins_pos) > 0 and len(case.p2b_edges) > 0:
        weighted_pins = [
            edge
            for edge in case.p2b_edges.tolist()
            if int(edge[1]) == block_index and int(edge[0]) != -1
        ]
        for pin_idx, _block, weight in sorted(
            weighted_pins, key=lambda item: item[2], reverse=True
        )[:2]:
            px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
            for dx, dy, source in [
                (0.0, 0.0, "pin_center"),
                (-width, 0.0, "pin_left"),
                (0.0, -height, "pin_below"),
            ]:
                candidates.append(
                    _candidate_with_metrics(
                        case,
                        state,
                        TypedAction(
                            ActionPrimitive.PLACE_ABSOLUTE,
                            block_index=block_index,
                            x=px + dx,
                            y=py + dy,
                            w=width,
                            h=height,
                            metadata={"source": source, "pin_weight": float(weight)},
                        ),
                        intent_type="pull_to_pin",
                    )
                )

    for target_index in state.placements:
        for dx, dy, relation in [
            (0.0, 0.0, "touch_right"),
            (0.0, height, "touch_top"),
            (-width, 0.0, "touch_left"),
            (0.0, -height, "touch_bottom"),
        ]:
            candidates.append(
                _candidate_with_metrics(
                    case,
                    state,
                    TypedAction(
                        ActionPrimitive.PLACE_RELATIVE,
                        block_index=block_index,
                        target_index=target_index,
                        dx=dx,
                        dy=dy,
                        w=width,
                        h=height,
                        metadata={"source": relation},
                    ),
                    intent_type="attach",
                )
            )

    return candidates


def _default_dims(case: FloorSetCase, block_index: int) -> tuple[float, float]:
    area = float(case.area_targets[block_index].item())
    if case.target_positions is not None:
        _x, _y, w, h = [float(v) for v in case.target_positions[block_index].tolist()]
        if w > 0 and h > 0:
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
    mode: CandidateMode = "strict",
    max_per_primitive: int | None = None,
) -> list[TypedAction]:
    remaining_blocks = remaining_blocks or [
        idx for idx in range(case.block_count) if idx not in state.placements
    ]
    candidates: list[TypedAction] = []
    placed_indices = list(state.placements.keys())

    for block_index in remaining_blocks:
        if mode in {"semantic", "relaxed"}:
            candidates.extend(_semantic_candidates_for_block(case, state, block_index))
        w, h = _default_dims(case, block_index)
        preplaced = False
        if case.target_positions is not None:
            tx, ty, tw, th = [float(v) for v in case.target_positions[block_index].tolist()]
            preplaced = (
                bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
                and tx >= 0
                and ty >= 0
                and tw > 0
                and th > 0
            )
            if preplaced:
                candidates.append(
                    TypedAction(
                        ActionPrimitive.PLACE_ABSOLUTE,
                        block_index=block_index,
                        x=tx,
                        y=ty,
                        w=tw,
                        h=th,
                        metadata={"source": "preplaced_anchor"},
                    )
                )
        candidates.append(
            TypedAction(
                ActionPrimitive.PLACE_ABSOLUTE,
                block_index=block_index,
                x=0.0,
                y=0.0,
                w=w,
                h=h,
                metadata={"source": "heuristic_origin"},
            )
        )

        boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
        if boundary_code != 0:
            candidates.append(
                TypedAction(
                    ActionPrimitive.PLACE_ABSOLUTE,
                    block_index=block_index,
                    x=0.0,
                    y=0.0,
                    w=w,
                    h=h,
                    metadata={"source": "boundary_seed"},
                )
            )

        for target_index in placed_indices:
            candidates.append(
                TypedAction(
                    ActionPrimitive.PLACE_RELATIVE,
                    block_index=block_index,
                    target_index=target_index,
                    dx=0.0,
                    dy=0.0,
                    w=w,
                    h=h,
                    metadata={"source": "adjacent_right"},
                )
            )
            candidates.append(
                TypedAction(
                    ActionPrimitive.PLACE_RELATIVE,
                    block_index=block_index,
                    target_index=target_index,
                    dx=0.0,
                    dy=h,
                    w=w,
                    h=h,
                    metadata={"source": "offset_vertical"},
                )
            )

    for block_index in placed_indices:
        if block_index in state.frozen_blocks:
            continue
        candidates.append(
            TypedAction(
                ActionPrimitive.FREEZE,
                block_index=block_index,
                metadata={"source": "freeze_after_place"},
            )
        )

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

    filtered = filter_legal_actions(case, state, candidates, mode=mode)
    if max_per_primitive is None:
        return filtered
    limited: list[TypedAction] = []
    per_primitive: dict[ActionPrimitive, int] = {}
    for candidate in filtered:
        count = per_primitive.get(candidate.primitive, 0)
        if count >= max_per_primitive:
            continue
        per_primitive[candidate.primitive] = count + 1
        limited.append(candidate)
    return limited


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


def actions_match(
    candidate: TypedAction,
    teacher: TypedAction,
    *,
    mode: CandidateMode = "strict",
) -> bool:
    if mode == "strict":
        return _same_action(candidate, teacher)
    if teacher.primitive is ActionPrimitive.FREEZE:
        return (
            candidate.primitive is ActionPrimitive.FREEZE
            and candidate.block_index == teacher.block_index
        )
    if candidate.block_index != teacher.block_index:
        return False
    teacher_width = teacher.w if teacher.w is not None else 0.0
    teacher_height = teacher.h if teacher.h is not None else 0.0
    candidate_width = candidate.w if candidate.w is not None else 0.0
    candidate_height = candidate.h if candidate.h is not None else 0.0
    same_shape = (
        abs(candidate_width - teacher_width) <= 1e-4
        and abs(candidate_height - teacher_height) <= 1e-4
    )
    return same_shape and candidate.primitive in {
        ActionPrimitive.PLACE_ABSOLUTE,
        ActionPrimitive.PLACE_RELATIVE,
        ActionPrimitive.ALIGN_BOUNDARY,
    }


def compute_expert_candidate_coverage(
    case: FloorSetCase, traces: list[PseudoTrace]
) -> CoverageReport:
    from puzzleplace.actions.executor import replay_actions

    total_steps = 0
    semantic_hits = 0
    relaxed_hits = 0
    strict_hits = 0
    teacher_hint_hits = 0

    for trace in traces:
        state = ExecutionState()
        for action in trace.actions:
            total_steps += 1
            remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
            semantic = generate_candidate_actions(
                case,
                state,
                remaining_blocks=remaining,
                mode="semantic",
            )
            relaxed = generate_candidate_actions(
                case,
                state,
                remaining_blocks=remaining,
                mode="relaxed",
            )
            strict = generate_candidate_actions(
                case,
                state,
                remaining_blocks=remaining,
                mode="strict",
            )
            teacher_hint = generate_candidate_actions(
                case,
                state,
                remaining_blocks=remaining,
                teacher_action=action,
                include_teacher_hint=True,
                mode="semantic",
            )
            if any(
                actions_match(candidate, action, mode="semantic")
                for candidate in semantic
            ):
                semantic_hits += 1
            if any(
                actions_match(candidate, action, mode="relaxed")
                for candidate in relaxed
            ):
                relaxed_hits += 1
            if any(actions_match(candidate, action, mode="strict") for candidate in strict):
                strict_hits += 1
            if any(
                actions_match(candidate, action, mode="strict")
                for candidate in teacher_hint
            ):
                teacher_hint_hits += 1
            replay_actions(case, [action], initial_state=state)

    return CoverageReport(
        total_steps=total_steps,
        semantic_hits=semantic_hits,
        relaxed_hits=relaxed_hits,
        strict_hits=strict_hits,
        teacher_hint_hits=teacher_hint_hits,
    )
