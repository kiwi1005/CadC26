"""Step7Q-D fresh-metric replay for finite operator expansions.

This sidecar consumes the Step7Q-C finite-action expansion deck and applies a
small deterministic geometry bridge against validation target geometry.  Learned
artifacts still contain no direct coordinates; coordinates are produced only
inside this replay/evaluation boundary.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import (
    Box,
    load_validation_cases,
    overlap_area,
)
from puzzleplace.experiments.step7l_learning_guided_replay import (
    nearest_nonoverlap_position as step7l_nearest_nonoverlap_position,
)
from puzzleplace.experiments.step7m_objective_corridor_replay import to_float
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_operator_learning import load_json, read_jsonl, write_json, write_jsonl

EPS = 1e-9
MEANINGFUL_COST_EPS = 1e-7
VALIDATION_LABEL_POLICY = "labels used for replay/evaluation only, not request generation"
REPLAY_SCOPE = "step7q_sidecar_finite_action_single_block_bridge"
ReplayEvaluator = Callable[[FloorSetCase, list[Box]], dict[str, Any]]


def replay_parameter_expansion_deck(
    base_dir: Path,
    examples_path: Path,
    expansion_deck_path: Path,
    parameter_summary_path: Path,
    replay_rows_out_path: Path,
    summary_out_path: Path,
    failures_out_path: Path,
    *,
    floorset_root: Path | None = None,
    max_candidates: int | None = None,
    auto_download: bool = False,
    slot_aware: bool = False,
    objective_aware_slot: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    examples = {row["example_id"]: row for row in read_jsonl(examples_path)}
    deck = read_jsonl(expansion_deck_path)
    if max_candidates is not None:
        deck = deck[:max_candidates]
    parameter_summary = load_json(parameter_summary_path)
    case_ids = sorted(
        {int(row["case_id"]) for row in deck if str(row.get("case_id", "")).isdigit()}
    )
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    evaluator = official_like_evaluator
    baseline_cache: dict[str, tuple[list[Box], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for request in deck:
        case_id = str(request.get("case_id"))
        case = cases.get(int(case_id)) if case_id.isdigit() else None
        if case is None:
            rows.append(missing_case_row(request))
            continue
        if case_id not in baseline_cache:
            baseline = positions_from_case_targets(case)
            baseline_cache[case_id] = (baseline, evaluator(case, baseline))
        baseline, before_eval = baseline_cache[case_id]
        parent = examples.get(str(request.get("parent_example_id")))
        rows.append(
            replay_expansion_row(
                request,
                parent,
                case,
                baseline,
                before_eval,
                evaluator,
                slot_aware=slot_aware,
                objective_aware_slot=objective_aware_slot,
            )
        )
    write_jsonl(replay_rows_out_path, rows)
    summary = summarize_fresh_replay(
        rows,
        parameter_summary,
        request_count=len(deck),
        expansion_deck_path=expansion_deck_path,
        replay_rows_path=replay_rows_out_path,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
    )
    write_json(summary_out_path, summary)
    write_json(failures_out_path, failures_by_case(rows))
    summary_out_path.with_suffix(".md").write_text(replay_markdown(summary), encoding="utf-8")
    return summary


def action_dict(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("operator_action")
    return action if isinstance(action, dict) else {}


def replay_expansion_row(
    request: dict[str, Any],
    parent: dict[str, Any] | None,
    case: FloorSetCase,
    baseline: list[Box],
    before_eval: dict[str, Any],
    evaluator: ReplayEvaluator,
    *,
    slot_aware: bool = False,
    objective_aware_slot: bool = False,
) -> dict[str, Any]:
    base = replay_base(request)
    action = action_dict(request)
    if not parent:
        return unrealized_row(base, request, "missing_parent_example", [])
    affected = affected_blocks_from_example(parent)
    if not affected:
        return unrealized_row(base, request, "no_affected_blocks", [])
    block_id = choose_block(case, affected, action)
    if block_id is None:
        return unrealized_row(base, request, "no_mutable_affected_block", [])
    target = target_box_for_action(case, baseline, block_id, action)
    if target is None:
        return unrealized_row(base, request, "unrealizable_action_direction", [block_id])
    if same_box(target, baseline[block_id]):
        return unrealized_row(base, request, "no_op_after_finite_action", [block_id])
    requested_target = target
    slot_adjustment: dict[str, Any] | None = None
    if overlaps_any(target, baseline, skip=block_id):
        if not slot_aware:
            return {
                **base,
                "candidate_id": f"{request.get('candidate_id')}:overlap",
                "block_id": block_id,
                "generation_status": "overlap_after_splice",
                "quality_gate_status": "infeasible_after_replay",
                "fresh_metric_available": False,
                "hard_feasible": False,
                "hard_feasible_nonnoop": False,
                "non_original_non_noop": True,
                "official_like_cost_improving": False,
                "metric_regressing": False,
                "moved_block_ids": [block_id],
                "moved_block_count": 1,
                "target_box": target,
                "slot_adjustment": slot_adjustment,
                "actual_objective_vector": None,
                "failure_attribution": "overlap_after_splice",
            }
        slot_choice = (
            objective_aware_nonoverlap_slot(
                case, baseline, before_eval, evaluator, block_id, target
            )
            if objective_aware_slot
            else slot_choice_from_box(nearest_nonoverlap_slot(baseline, block_id, target))
        )
        if slot_choice is None:
            return unrealized_row(base, request, "no_feasible_slot_after_overlap", [block_id])
        target = slot_choice["slot_target_box"]
        slot_adjustment = {
            "slot_aware": True,
            "objective_aware_slot": objective_aware_slot,
            "requested_target_box": requested_target,
            **slot_choice,
        }
    after = list(baseline)
    after[block_id] = target
    after_eval = evaluator(case, after)
    legality = summarize_hard_legality(case, after)
    delta = actual_delta(before_eval, after_eval)
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    status = quality_gate_status(delta, hard_feasible)
    strict = strict_meaningful_winner(delta, hard_feasible)
    return {
        **base,
        "candidate_id": f"{request.get('candidate_id')}:step7q:fresh_metric_bridge",
        "block_id": block_id,
        "generation_status": "realized_slot_aware_finite_action_bridge"
        if slot_adjustment
        else "realized_finite_action_single_block_bridge",
        "quality_gate_status": status,
        "fresh_metric_available": True,
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible,
        "non_original_non_noop": True,
        "official_like_cost_improving": delta["official_like_cost_delta"] < -EPS,
        "metric_regressing": delta["official_like_cost_delta"] > EPS,
        "moved_block_ids": [block_id],
        "moved_block_count": 1,
        "target_box": target,
        "slot_adjustment": slot_adjustment,
        "actual_objective_vector": delta,
        **delta,
        "actual_all_vector_nonregressing": all_vector_nonregressing(delta, hard_feasible),
        "strict_meaningful_winner": strict,
        "official_before_quality": before_eval["quality"],
        "official_after_quality": after_eval["quality"],
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "failure_attribution": failure_attribution(status, delta, legality.overlap_violations),
        "replayed_signature": replayed_signature(str(request.get("case_id")), block_id, target),
    }


def replay_base(request: dict[str, Any]) -> dict[str, Any]:
    action = action_dict(request)
    return {
        "schema": "step7q_fresh_metric_replay_row_v1",
        "request_id": request.get("candidate_id"),
        "case_id": str(request.get("case_id")),
        "parent_example_id": request.get("parent_example_id"),
        "source_subproblem_id": request.get("source_subproblem_id"),
        "source_candidate_id": request.get("source_candidate_id"),
        "operator_family": action.get("operator_family"),
        "direction_bin": action.get("direction_bin"),
        "magnitude_bin": action.get("magnitude_bin"),
        "bbox_guard_mode": action.get("bbox_guard_mode"),
        "vector_guard_mode": action.get("vector_guard_mode"),
        "replay_scope": REPLAY_SCOPE,
        "validation_label_policy": VALIDATION_LABEL_POLICY,
        "request_score": request.get("expansion_score"),
    }


def unrealized_row(
    base: dict[str, Any], request: dict[str, Any], status: str, moved: list[int]
) -> dict[str, Any]:
    return {
        **base,
        "candidate_id": f"{request.get('candidate_id')}:unrealized",
        "block_id": moved[0] if moved else None,
        "generation_status": status,
        "quality_gate_status": status,
        "fresh_metric_available": False,
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "non_original_non_noop": False,
        "official_like_cost_improving": False,
        "metric_regressing": False,
        "moved_block_ids": moved,
        "moved_block_count": len(moved),
        "actual_objective_vector": None,
        "failure_attribution": status,
    }


def missing_case_row(request: dict[str, Any]) -> dict[str, Any]:
    return unrealized_row(replay_base(request), request, "missing_case", [])


def affected_blocks_from_example(example: dict[str, Any]) -> list[int]:
    graph_value = example.get("graph")
    graph: dict[str, Any] = graph_value if isinstance(graph_value, dict) else {}
    blocks = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("type") != "affected_block":
            continue
        dst = str(edge.get("dst", ""))
        try:
            blocks.append(int(dst.rsplit(":", 1)[-1]))
        except ValueError:
            continue
    return sorted(set(blocks))


def choose_block(case: FloorSetCase, affected: list[int], action: dict[str, Any]) -> int | None:
    mutable = [
        block for block in affected if 0 <= block < case.block_count and is_mutable(case, block)
    ]
    if not mutable:
        return None
    direction = str(action.get("direction_bin"))
    positions = positions_from_case_targets(case)
    cx = center_x(positions)
    cy = center_y(positions)
    if direction == "bbox_shrink_x":
        return max(
            mutable,
            key=lambda block: abs((positions[block][0] + positions[block][2] / 2.0) - cx),
        )
    if direction == "bbox_shrink_y":
        return max(
            mutable,
            key=lambda block: abs((positions[block][1] + positions[block][3] / 2.0) - cy),
        )
    if direction == "hpwl_sink_toward_pins":
        return max(mutable, key=lambda block: terminal_weight(case, block))
    return mutable[0]


def target_box_for_action(
    case: FloorSetCase, positions: list[Box], block_id: int, action: dict[str, Any]
) -> Box | None:
    x, y, w, h = positions[block_id]
    scale = magnitude_scale(str(action.get("magnitude_bin")))
    step = max(min(w, h) * scale, 1e-4)
    direction = str(action.get("direction_bin"))
    dx = 0.0
    dy = 0.0
    if direction == "hpwl_sink_toward_pins":
        centroid = terminal_centroid(case, block_id)
        if centroid is None:
            return None
        bx = x + w / 2.0
        by = y + h / 2.0
        dx = clamp(centroid[0] - bx, -step, step)
        dy = clamp(centroid[1] - by, -step, step)
    elif direction == "bbox_shrink_x":
        dx = step if x + w / 2.0 < center_x(positions) else -step
    elif direction == "bbox_shrink_y":
        dy = step if y + h / 2.0 < center_y(positions) else -step
    elif direction == "slack_fill_left":
        dx = -step
    elif direction == "slack_fill_right":
        dx = step
    elif direction == "slack_fill_up":
        dy = step
    elif direction == "slack_fill_down":
        dy = -step
    elif direction == "soft_release":
        dx, dy = soft_release_delta(case, block_id, step)
    elif direction == "blocker_unblock":
        dy = step
    else:
        return None
    return (x + dx, y + dy, w, h)


def slot_choice_from_box(slot: Box | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    return {
        "slot_target_box": slot,
        "slot_distance": 0.0,
        "slot_candidate_count": 1,
        "slot_selection_score": None,
    }


def objective_aware_nonoverlap_slot(
    case: FloorSetCase,
    positions: list[Box],
    before_eval: dict[str, Any],
    evaluator: ReplayEvaluator,
    block_id: int,
    target: Box,
) -> dict[str, Any] | None:
    target_center = box_center(target)
    candidates = nonoverlap_slot_candidates(positions, block_id, target, limit=24)
    if not candidates:
        return None
    scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for slot in candidates:
        after = list(positions)
        after[block_id] = slot
        after_eval = evaluator(case, after)
        legality = summarize_hard_legality(case, after)
        hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
        delta = actual_delta(before_eval, after_eval)
        key = slot_objective_key(delta, hard_feasible, slot, target_center)
        scored.append(
            (
                key,
                {
                    "slot_target_box": slot,
                    "slot_distance": center_distance(slot, target_center),
                    "slot_candidate_count": len(candidates),
                    "slot_selection_score": list(key),
                    "slot_predicted_objective_vector": delta,
                    "slot_predicted_hard_feasible": hard_feasible,
                },
            )
        )
    scored.sort(key=lambda row: row[0])
    return scored[0][1]


def slot_objective_key(
    delta: dict[str, float], hard_feasible: bool, slot: Box, target_center: tuple[float, float]
) -> tuple[float, ...]:
    return (
        0.0 if hard_feasible else 1.0,
        0.0 if all_vector_nonregressing(delta, hard_feasible) else 1.0,
        1.0 if delta["soft_constraint_delta"] > EPS else 0.0,
        1.0 if delta["bbox_area_delta"] > EPS else 0.0,
        1.0 if delta["hpwl_delta"] > EPS else 0.0,
        max(delta["soft_constraint_delta"], 0.0),
        max(delta["bbox_area_delta"], 0.0),
        max(delta["hpwl_delta"], 0.0),
        delta["official_like_cost_delta"],
        center_distance(slot, target_center),
    )


def nonoverlap_slot_candidates(
    positions: list[Box], block_id: int, target: Box, *, limit: int | None = None
) -> list[Box]:
    current = positions[block_id]
    _x, _y, w, h = current
    target_center = box_center(target)
    frame = frame_from_positions(positions)
    step_x = max(w, target[2], 1.0)
    step_y = max(h, target[3], 1.0)
    proposals: list[Box] = []
    base_x = target_center[0] - w / 2.0
    base_y = target_center[1] - h / 2.0
    for radius in range(0, 7):
        for dy_i in range(-radius, radius + 1):
            for dx_i in range(-radius, radius + 1):
                if radius and max(abs(dx_i), abs(dy_i)) != radius:
                    continue
                proposals.append((base_x + dx_i * step_x, base_y + dy_i * step_y, w, h))
    proposals.extend(frame_corner_slots(frame, w, h))
    proposals.extend(adjacent_obstacle_slots(positions, block_id, target))
    deduped: dict[tuple[int, int, int, int], Box] = {}
    for slot in proposals:
        if same_box(slot, current):
            continue
        if overlaps_any(slot, positions, skip=block_id):
            continue
        deduped[rounded_box_key(slot)] = slot
    candidates = list(deduped.values())
    candidates.sort(key=lambda slot: slot_prefilter_key(slot, positions[block_id], target))
    return candidates if limit is None else candidates[:limit]


def slot_prefilter_key(slot: Box, current: Box, target: Box) -> tuple[float, float, float, float]:
    target_center = box_center(target)
    current_center = box_center(current)
    return (
        center_distance(slot, current_center),
        center_distance(slot, target_center),
        abs((slot[2] * slot[3]) - (current[2] * current[3])),
        slot[0] + slot[1],
    )


def frame_corner_slots(frame: dict[str, float], w: float, h: float) -> list[Box]:
    x = frame["x"]
    y = frame["y"]
    fw = frame["w"]
    fh = frame["h"]
    return [
        (x, y, w, h),
        (x + max(fw - w, 0.0), y, w, h),
        (x, y + max(fh - h, 0.0), w, h),
        (x + max(fw - w, 0.0), y + max(fh - h, 0.0), w, h),
    ]


def adjacent_obstacle_slots(positions: list[Box], block_id: int, target: Box) -> list[Box]:
    _x, _y, w, h = positions[block_id]
    tx, ty, _tw, _th = target
    out: list[Box] = []
    for index, (ox, oy, ow, oh) in enumerate(positions):
        if index == block_id:
            continue
        out.extend(
            [
                (ox - w, ty, w, h),
                (ox + ow, ty, w, h),
                (tx, oy - h, w, h),
                (tx, oy + oh, w, h),
                (ox - w, oy - h, w, h),
                (ox + ow, oy + oh, w, h),
            ]
        )
    return out


def rounded_box_key(box: Box) -> tuple[int, int, int, int]:
    return tuple(int(round(value * 1000.0)) for value in box)  # type: ignore[return-value]


def nearest_nonoverlap_slot(positions: list[Box], block_id: int, target: Box) -> Box | None:
    frame = frame_from_positions(positions)
    target_center = (target[0] + target[2] / 2.0, target[1] + target[3] / 2.0)
    window = {
        "cx": target_center[0],
        "cy": target_center[1],
        "w": target[2],
        "h": target[3],
        "frame": frame,
    }
    return step7l_nearest_nonoverlap_position(
        positions, block_id, target_center=target_center, window=window
    )


def frame_from_positions(positions: list[Box]) -> dict[str, float]:
    x0 = min(x for x, _, _, _ in positions)
    y0 = min(y for _, y, _, _ in positions)
    x1 = max(x + w for x, _, w, _ in positions)
    y1 = max(y + h for _, y, _, h in positions)
    span = max(x1 - x0, y1 - y0, 1.0)
    margin = span * 0.05
    return {
        "x": x0 - margin,
        "y": y0 - margin,
        "w": x1 - x0 + 2 * margin,
        "h": y1 - y0 + 2 * margin,
    }


def box_center(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def center_distance(box: Box, target_center: tuple[float, float]) -> float:
    cx, cy = box_center(box)
    return ((cx - target_center[0]) ** 2 + (cy - target_center[1]) ** 2) ** 0.5


def is_mutable(case: FloorSetCase, block_id: int) -> bool:
    fixed = bool(case.constraints[block_id, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[block_id, ConstraintColumns.PREPLACED].item())
    return not fixed and not preplaced


def magnitude_scale(value: str) -> float:
    return {"tiny": 0.025, "small": 0.05, "medium": 0.10, "slack_limited": 0.05}.get(
        value, 0.05
    )


def terminal_centroid(case: FloorSetCase, block_id: int) -> tuple[float, float] | None:
    total_w = 0.0
    sx = 0.0
    sy = 0.0
    for pin_idx, b_idx, weight in case.p2b_edges.tolist():
        if int(b_idx) != block_id or int(pin_idx) < 0:
            continue
        pin = int(pin_idx)
        if pin >= len(case.pins_pos):
            continue
        w = float(weight)
        px, py = [float(v) for v in case.pins_pos[pin].tolist()]
        sx += px * w
        sy += py * w
        total_w += w
    if total_w <= 0.0:
        return None
    return sx / total_w, sy / total_w


def terminal_weight(case: FloorSetCase, block_id: int) -> float:
    total = 0.0
    for pin_idx, b_idx, weight in case.p2b_edges.tolist():
        if int(b_idx) == block_id and int(pin_idx) >= 0:
            total += float(weight)
    return total


def soft_release_delta(case: FloorSetCase, block_id: int, step: float) -> tuple[float, float]:
    boundary = int(case.constraints[block_id, ConstraintColumns.BOUNDARY].item())
    if boundary == 1:
        return step, 0.0
    if boundary == 2:
        return -step, 0.0
    if boundary == 4:
        return 0.0, -step
    if boundary == 8:
        return 0.0, step
    return step, 0.0


def summarize_fresh_replay(
    rows: list[dict[str, Any]],
    parameter_summary: dict[str, Any],
    *,
    request_count: int,
    expansion_deck_path: Path,
    replay_rows_path: Path,
    runtime_proxy_ms: float,
) -> dict[str, Any]:
    fresh = [row for row in rows if row.get("fresh_metric_available")]
    hard = [row for row in fresh if row.get("hard_feasible_nonnoop")]
    strict = [row for row in hard if row.get("strict_meaningful_winner")]
    overlap_count = sum(
        int(str(row.get("failure_attribution")) == "overlap_after_splice") for row in rows
    )
    soft = sum(int(to_float(row.get("soft_constraint_delta")) > EPS) for row in hard)
    bbox = sum(int(to_float(row.get("bbox_area_delta")) > EPS) for row in hard)
    hpwl = sum(int(to_float(row.get("hpwl_delta")) > EPS) for row in hard)
    all_vector = sum(int(bool(row.get("actual_all_vector_nonregressing"))) for row in hard)
    represented_cases = {str(row.get("case_id")) for row in rows}
    unique_signatures = {
        row.get("replayed_signature") for row in hard if row.get("replayed_signature")
    }
    phase4_gate_open = (
        request_count >= 96
        and len(represented_cases) >= 8
        and len(hard) >= 60
        and overlap_count == 0
        and all_vector >= 30
        and soft / max(request_count, 1) <= 0.10
        and bbox / max(request_count, 1) <= 0.10
        and len(strict) >= 3
        and sum(int(str(row.get("case_id")) not in {"24", "25"}) for row in strict) >= 1
        and len({row.get("source_candidate_id") for row in strict}) >= 3
    )
    risk_replay_gate = (
        request_count >= 96
        and len(represented_cases) >= 8
        and overlap_count < 23
        and soft / max(request_count, 1) <= 0.052083333333333336
        and bbox / max(request_count, 1) == 0.0
    )
    if phase4_gate_open:
        decision = "promote_to_phase4_ablation_review"
        allowed_next_phase: str | None = "step7q_phase4_ablation_review"
        next_recommendation = "review_strict_winners_before_phase4"
    elif fresh:
        decision = "fresh_replay_executed_strict_gate_closed"
        allowed_next_phase = None
        next_recommendation = "inspect_failures_then_refine_executor_or_parameter_deck"
    else:
        decision = "stop_fresh_replay_no_realized_metrics"
        allowed_next_phase = None
        next_recommendation = "fix_geometry_bridge_before_more_learning"
    return {
        "schema": "step7q_fresh_metric_replay_summary_v1",
        "decision": decision,
        "expansion_deck_path": str(expansion_deck_path),
        "replay_rows_path": str(replay_rows_path),
        "parameter_expansion_decision": parameter_summary.get("decision"),
        "request_count": request_count,
        "replay_row_count": len(rows),
        "fresh_metric_available_count": len(fresh),
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "overlap_after_splice_count": overlap_count,
        "soft_regression_count": soft,
        "soft_regression_rate": soft / max(request_count, 1),
        "bbox_regression_count": bbox,
        "bbox_regression_rate": bbox / max(request_count, 1),
        "hpwl_regression_count": hpwl,
        "hpwl_regression_rate": hpwl / max(request_count, 1),
        "actual_all_vector_nonregressing_count": all_vector,
        "strict_meaningful_winner_count": len(strict),
        "strict_winner_case_counts": dict(Counter(str(row.get("case_id")) for row in strict)),
        "non_case024_non_case025_strict_winner_count": sum(
            int(str(row.get("case_id")) not in {"24", "25"}) for row in strict
        ),
        "represented_case_count": len(represented_cases),
        "unique_replayed_signature_count": len(unique_signatures),
        "status_counts": dict(Counter(str(row.get("quality_gate_status")) for row in rows)),
        "failure_counts": dict(Counter(str(row.get("failure_attribution")) for row in rows)),
        "per_case_status_counts": per_case_status_counts(rows),
        "risk_replay_gate_open": risk_replay_gate,
        "phase4_gate_open": phase4_gate_open,
        "allowed_next_phase": allowed_next_phase,
        "next_recommendation": next_recommendation,
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_candidate_ms": runtime_proxy_ms / max(request_count, 1),
        "objective_aware_slot_replay": any(
            bool(row.get("slot_adjustment", {}).get("objective_aware_slot"))
            for row in rows
            if row.get("fresh_metric_available")
        ),
        "slot_aware_replay": any(
            bool(row.get("slot_adjustment")) for row in rows if row.get("fresh_metric_available")
        ),
        "slot_adjusted_count": sum(
            int(bool(row.get("slot_adjustment")))
            for row in rows
            if row.get("fresh_metric_available")
        ),
        "replay_contract": (
            "finite operator actions are converted to geometry only inside this sidecar "
            "fresh-metric boundary"
        ),
    }


def failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("failure_attribution"))] += 1
    return {
        "schema": "step7q_fresh_metric_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def per_case_status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("quality_gate_status"))] += 1
    return {case_id: dict(counter) for case_id, counter in sorted(by_case.items())}


def replay_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7Q Fresh Metric Replay Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- request_count: {summary['request_count']}",
            f"- fresh_metric_available_count: {summary['fresh_metric_available_count']}",
            f"- fresh_hard_feasible_nonnoop_count: {summary['fresh_hard_feasible_nonnoop_count']}",
            f"- overlap_after_splice_count: {summary['overlap_after_splice_count']}",
            f"- soft_regression_rate: {summary['soft_regression_rate']}",
            f"- bbox_regression_rate: {summary['bbox_regression_rate']}",
            f"- actual_all_vector_nonregressing_count: "
            f"{summary['actual_all_vector_nonregressing_count']}",
            f"- strict_meaningful_winner_count: {summary['strict_meaningful_winner_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            f"- unique_replayed_signature_count: {summary['unique_replayed_signature_count']}",
            f"- risk_replay_gate_open: {summary['risk_replay_gate_open']}",
            f"- phase4_gate_open: {summary['phase4_gate_open']}",
            f"- allowed_next_phase: {summary['allowed_next_phase']}",
            "",
        ]
    )


def actual_delta(before_eval: dict[str, Any], after_eval: dict[str, Any]) -> dict[str, float]:
    before_q = before_eval["quality"]
    after_q = after_eval["quality"]
    return {
        "official_like_cost_delta": to_float(after_q.get("cost")) - to_float(before_q.get("cost")),
        "hpwl_delta": to_float(after_q.get("HPWLgap")) - to_float(before_q.get("HPWLgap")),
        "bbox_area_delta": to_float(after_q.get("Areagap_bbox"))
        - to_float(before_q.get("Areagap_bbox")),
        "soft_constraint_delta": to_float(after_q.get("Violationsrelative"))
        - to_float(before_q.get("Violationsrelative")),
    }


def quality_gate_status(delta: dict[str, float], hard_feasible: bool) -> str:
    if not hard_feasible:
        return "infeasible_after_replay"
    if strict_meaningful_winner(delta, hard_feasible):
        return "strict_meaningful_winner"
    if delta["official_like_cost_delta"] > EPS:
        return "metric_regression"
    return "metric_tradeoff_report_only"


def strict_meaningful_winner(delta: dict[str, float], hard_feasible: bool) -> bool:
    return (
        hard_feasible
        and delta["official_like_cost_delta"] < -MEANINGFUL_COST_EPS
        and delta["hpwl_delta"] <= EPS
        and delta["bbox_area_delta"] <= EPS
        and delta["soft_constraint_delta"] <= EPS
    )


def all_vector_nonregressing(delta: dict[str, float], hard_feasible: bool) -> bool:
    return (
        hard_feasible
        and delta["hpwl_delta"] <= EPS
        and delta["bbox_area_delta"] <= EPS
        and delta["soft_constraint_delta"] <= EPS
    )


def failure_attribution(status: str, delta: dict[str, float], overlap_violations: int) -> str:
    if status == "strict_meaningful_winner":
        return "none"
    if status == "infeasible_after_replay":
        return "overlap_after_splice" if overlap_violations > 0 else "hard_infeasible_after_replay"
    if delta["soft_constraint_delta"] > EPS:
        return "soft_regression"
    if delta["bbox_area_delta"] > EPS:
        return "bbox_regression"
    if delta["hpwl_delta"] > EPS:
        return "hpwl_regression"
    return "metric_tradeoff"


def overlaps_any(candidate: Box, positions: list[Box], *, skip: int) -> bool:
    return any(
        index != skip and overlap_area(candidate, other) > EPS
        for index, other in enumerate(positions)
    )


def same_box(a: Box, b: Box) -> bool:
    return all(abs(left - right) <= EPS for left, right in zip(a, b, strict=False))


def center_x(positions: list[Box]) -> float:
    return (min(x for x, _, _, _ in positions) + max(x + w for x, _, w, _ in positions)) / 2.0


def center_y(positions: list[Box]) -> float:
    return (min(y for _, y, _, _ in positions) + max(y + h for _, y, _, h in positions)) / 2.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def replayed_signature(case_id: str, block_id: int, target: Box) -> str:
    x, y, w, h = target
    return f"case={case_id}|block={block_id}|x={x:.3f}|y={y:.3f}|w={w:.3f}|h={h:.3f}"


def official_like_evaluator(case: FloorSetCase, positions: list[Box]) -> dict[str, Any]:
    return evaluate_positions(case, positions, runtime=1.0)


__all__ = [
    "affected_blocks_from_example",
    "replay_expansion_row",
    "replay_parameter_expansion_deck",
    "summarize_fresh_replay",
    "target_box_for_action",
]
