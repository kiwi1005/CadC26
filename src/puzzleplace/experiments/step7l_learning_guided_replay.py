"""Step7L sidecar replay bridge for learning-guided target requests.

The bridge is intentionally conservative: Step7L requests are generated from
visible validation inputs only, while this module consumes official validation
labels only at replay/evaluation time to build the baseline geometry that the
existing official-like evaluator requires. It does not modify contest runtime or
finalizer code.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.data.floorset_adapter import adapt_validation_batch
from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.floorset_training_corpus import (
    _auto_yes_download,
    _import_official_evaluator,
    resolve_floorset_root,
    write_json,
)

Box = tuple[float, float, float, float]
EPS = 1e-9
ReplayEvaluator = Callable[[FloorSetCase, list[Box]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReplayAttempt:
    status: str
    positions: list[Box] | None
    moved_block_ids: list[int]
    notes: list[str]
    target_distance: float | None = None
    displacement: float | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def replay_requests(
    base_dir: Path,
    requests_path: Path,
    replay_rows_path: Path,
    summary_path: Path,
    failures_path: Path,
    *,
    floorset_root: Path | None = None,
    max_requests: int | None = None,
    auto_download: bool = False,
) -> dict[str, Any]:
    """Replay Step7L request windows with a bounded obstacle-aware bridge."""

    started = time.perf_counter()
    requests = read_jsonl(requests_path)
    if max_requests is not None:
        anchors = [row for row in requests if row.get("is_anchor")]
        non_anchors = [row for row in requests if not row.get("is_anchor")][:max_requests]
        requests = anchors + non_anchors
    case_ids = sorted(
        {int(row["loader_index"]) for row in requests if row.get("loader_index") is not None}
    )
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    evaluator = _official_like_evaluator
    replay_rows: list[dict[str, Any]] = []
    baseline_cache: dict[str, tuple[list[Box], dict[str, Any]]] = {}
    for request in requests:
        case_key = str(request.get("case_id"))
        case = cases.get(int(request.get("loader_index", -1)))
        if case is None:
            replay_rows.append(_missing_case_row(request))
            continue
        if case_key not in baseline_cache:
            baseline = positions_from_case_targets(case)
            baseline_cache[case_key] = (baseline, evaluator(case, baseline))
        positions, before_eval = baseline_cache[case_key]
        replay_rows.append(replay_request_row(request, case, positions, before_eval, evaluator))

    write_jsonl(replay_rows_path, replay_rows)
    summary = summarize_replay_rows(
        replay_rows,
        request_count=len(requests),
        request_path=requests_path,
        replay_rows_path=replay_rows_path,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
    )
    write_json(summary_path, summary)
    write_json(failures_path, failures_by_case(replay_rows))
    summary_path.with_name("step7l_phase2_decision.md").write_text(
        decision_markdown(summary), encoding="utf-8"
    )
    return summary


def load_validation_cases(
    base_dir: Path,
    case_ids: list[int],
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
) -> dict[int, FloorSetCase]:
    resolved = resolve_floorset_root(base_dir, floorset_root)
    if resolved is None:
        raise FileNotFoundError("Could not resolve external/FloorSet official checkout")
    evaluator = _import_official_evaluator(resolved)
    requested = set(case_ids)
    cases: dict[int, FloorSetCase] = {}
    with _auto_yes_download(auto_download):
        loader = evaluator.get_validation_dataloader(data_path=str(resolved), batch_size=1)
        for index, batch in enumerate(loader):
            if index > max(requested, default=-1):
                break
            if index not in requested:
                continue
            cases[index] = adapt_validation_batch(
                batch,
                case_id=index,
                raw={
                    "loader_index": index,
                    "label_policy": "validation labels used only for replay evaluation",
                },
            )
    return cases


def replay_request_row(
    request: dict[str, Any],
    case: FloorSetCase,
    baseline_positions: list[Box],
    before_eval: dict[str, Any],
    evaluator: ReplayEvaluator,
) -> dict[str, Any]:
    base = {
        "schema": "step7l_phase2_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": request.get("case_id"),
        "loader_index": request.get("loader_index"),
        "source_family": request.get("source_family"),
        "block_id": request.get("block_id"),
        "move_family": request.get("move_family"),
        "request_route_class": request.get("route_class"),
        "heatmap_score": request.get("heatmap_score"),
        "is_anchor": bool(request.get("is_anchor")),
        "request_global_report_only": bool(request.get("global_report_only")),
        "replay_scope": "sidecar_obstacle_aware_single_block_bridge",
        "validation_label_policy": "labels used for replay/evaluation only, not request generation",
    }
    if request.get("is_anchor"):
        return {
            **base,
            "candidate_id": f"{request.get('request_id')}:replay_anchor",
            "generation_status": "original_anchor",
            "quality_gate_status": "original_anchor",
            "fresh_metric_available": True,
            "non_original_non_noop": False,
            "hard_feasible": bool(before_eval["quality"].get("feasible")),
            "hard_feasible_nonnoop": False,
            "official_like_cost_improving": False,
            "dominated_by_original": False,
            "metric_regressing": False,
            "moved_block_ids": [],
            "moved_block_count": 0,
            "actual_route_prediction": None,
            "objective_vector": _zero_delta(),
            "official_like_cost_delta": 0.0,
            "hpwl_delta": 0.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "failure_attribution": "original_anchor",
        }

    attempt = obstacle_aware_single_block_attempt(request, case, baseline_positions)
    if attempt.positions is None:
        return {
            **base,
            "candidate_id": f"{request.get('request_id')}:unrealized",
            "generation_status": attempt.status,
            "quality_gate_status": attempt.status,
            "fresh_metric_available": False,
            "non_original_non_noop": False,
            "hard_feasible": False,
            "hard_feasible_nonnoop": False,
            "official_like_cost_improving": False,
            "dominated_by_original": True,
            "metric_regressing": False,
            "moved_block_ids": attempt.moved_block_ids,
            "moved_block_count": len(attempt.moved_block_ids),
            "actual_route_prediction": None,
            "objective_vector": None,
            "failure_attribution": attempt.status,
            "application": _attempt_json(attempt),
        }

    after_eval = evaluator(case, attempt.positions)
    legality = summarize_hard_legality(case, attempt.positions)
    delta = delta_metrics(before_eval, after_eval)
    no_op = len(attempt.moved_block_ids) == 0
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    dominated = dominated_by_original(delta, hard_feasible)
    status = quality_gate_status(delta, hard_feasible, no_op)
    route_prediction = route_prediction_for_attempt(
        case, baseline_positions, attempt.positions, hard_feasible
    )
    metric_regressing = delta["official_like_cost_delta"] > EPS
    row = {
        **base,
        "candidate_id": f"{request.get('request_id')}:step7l:single_block_bridge",
        "generation_status": attempt.status,
        "quality_gate_status": status,
        "fresh_metric_available": True,
        "non_original_non_noop": not no_op,
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible and not no_op,
        "official_like_cost_improving": delta["official_like_cost_delta"] < -EPS,
        "dominated_by_original": dominated,
        "metric_regressing": metric_regressing,
        "moved_block_ids": attempt.moved_block_ids,
        "moved_block_count": len(attempt.moved_block_ids),
        "actual_route_prediction": route_prediction,
        "actual_locality_class": route_prediction["predicted_locality_class"],
        "actual_repair_mode": route_prediction["predicted_repair_mode"],
        "objective_vector": delta,
        **delta,
        "official_after_quality": after_eval["quality"],
        "official_before_quality": before_eval["quality"],
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "failure_attribution": failure_attribution(status, delta, legality.overlap_violations),
        "application": _attempt_json(attempt),
        "replayed_signature": replayed_signature(
            str(request.get("case_id")),
            request.get("source_family"),
            request.get("block_id"),
            attempt.positions,
        ),
    }
    return row


def obstacle_aware_single_block_attempt(
    request: dict[str, Any], case: FloorSetCase, positions: list[Box]
) -> ReplayAttempt:
    block_id = request.get("block_id")
    if not isinstance(block_id, int):
        return ReplayAttempt("invalid_request", None, [], ["missing integer block_id"])
    if block_id < 0 or block_id >= case.block_count:
        return ReplayAttempt("invalid_request", None, [], ["block_id outside case"])
    if not is_movable(case, block_id):
        return ReplayAttempt("no_movable_blocks", None, [], ["fixed/preplaced block"])
    window = request.get("target_window")
    if not isinstance(window, dict):
        return ReplayAttempt("invalid_request", None, [], ["missing target_window"])
    current = positions[block_id]
    target_cx = _float(window.get("cx"), current[0] + current[2] / 2.0)
    target_cy = _float(window.get("cy"), current[1] + current[3] / 2.0)
    candidate = nearest_nonoverlap_position(
        positions,
        block_id,
        target_center=(target_cx, target_cy),
        window=window,
    )
    if candidate is None:
        return ReplayAttempt("no_feasible_space", None, [], ["no full-case non-overlap slot found"])
    new_positions = list(positions)
    new_positions[block_id] = candidate
    moved = _moved_blocks(positions, new_positions)
    if not moved:
        return ReplayAttempt("no_op_report_only", None, [], ["best feasible slot is original"])
    target_distance = _center_distance(candidate, (target_cx, target_cy))
    displacement = abs(candidate[0] - current[0]) + abs(candidate[1] - current[1])
    return ReplayAttempt(
        "realized_single_block_target_window",
        new_positions,
        moved,
        ["nearest full-case non-overlap slot to target window"],
        target_distance=target_distance,
        displacement=displacement,
    )


def nearest_nonoverlap_position(
    positions: list[Box],
    block_id: int,
    *,
    target_center: tuple[float, float],
    window: dict[str, Any],
) -> Box | None:
    current = positions[block_id]
    _x, _y, w, h = current
    step_x = max(_float(window.get("w"), w), w / 2.0, 1.0)
    step_y = max(_float(window.get("h"), h), h / 2.0, 1.0)
    base_x = target_center[0] - w / 2.0
    base_y = target_center[1] - h / 2.0
    proposals: list[Box] = []
    for radius in range(0, 5):
        for dy_i in range(-radius, radius + 1):
            for dx_i in range(-radius, radius + 1):
                if radius and max(abs(dx_i), abs(dy_i)) != radius:
                    continue
                proposals.append((base_x + dx_i * step_x, base_y + dy_i * step_y, w, h))
    proposals.extend(_frame_corner_slots(window, w, h))
    feasible: list[tuple[float, float, Box]] = []
    current_center = _box_center(current)
    for candidate in proposals:
        if _same_box(candidate, current):
            continue
        if overlaps_any(candidate, positions, skip=block_id):
            continue
        feasible.append(
            (
                _center_distance(candidate, target_center),
                math.hypot(
                    _box_center(candidate)[0] - current_center[0],
                    _box_center(candidate)[1] - current_center[1],
                ),
                candidate,
            )
        )
    feasible.sort(key=lambda row: (row[0], row[1], row[2][1], row[2][0]))
    return feasible[0][2] if feasible else None


def _frame_corner_slots(window: dict[str, Any], w: float, h: float) -> list[Box]:
    frame = window.get("frame")
    if not isinstance(frame, dict):
        return []
    x = _float(frame.get("x"))
    y = _float(frame.get("y"))
    fw = _float(frame.get("w"))
    fh = _float(frame.get("h"))
    if fw <= 0 or fh <= 0:
        return []
    return [
        (x, y, w, h),
        (x + max(fw - w, 0.0), y, w, h),
        (x, y + max(fh - h, 0.0), w, h),
        (x + max(fw - w, 0.0), y + max(fh - h, 0.0), w, h),
    ]


def overlaps_any(candidate: Box, positions: list[Box], *, skip: int) -> bool:
    return any(
        index != skip and overlap_area(candidate, other) > EPS
        for index, other in enumerate(positions)
    )


def overlap_area(a: Box, b: Box) -> float:
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def is_movable(case: FloorSetCase, block_id: int) -> bool:
    fixed = bool(case.constraints[block_id, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[block_id, ConstraintColumns.PREPLACED].item())
    return not fixed and not preplaced


def route_prediction_for_attempt(
    case: FloorSetCase,
    before: list[Box],
    after: list[Box],
    hard_feasible: bool,
) -> dict[str, Any]:
    moved = _moved_blocks(before, after)
    return predict_move_locality(
        case_id=int(case.case_id),
        block_count=case.block_count,
        changed_block_count=len(moved),
        touched_region_count=1 if len(moved) <= 1 else 2,
        macro_closure_size=max(len(moved), 1),
        min_region_slack=0.0,
        free_space_fit_ratio=1.0,
        hard_summary={"hard_feasible": hard_feasible},
    )


def delta_metrics(before_eval: dict[str, Any], after_eval: dict[str, Any]) -> dict[str, float]:
    before_q = before_eval["quality"]
    after_q = after_eval["quality"]
    return {
        "official_like_cost_delta": _float(after_q.get("cost")) - _float(before_q.get("cost")),
        "hpwl_delta": _float(after_q.get("HPWLgap")) - _float(before_q.get("HPWLgap")),
        "bbox_area_delta": _float(after_q.get("Areagap_bbox"))
        - _float(before_q.get("Areagap_bbox")),
        "soft_constraint_delta": _float(after_q.get("Violationsrelative"))
        - _float(before_q.get("Violationsrelative")),
    }


def dominated_by_original(delta: dict[str, float], hard_feasible: bool) -> bool:
    if not hard_feasible:
        return True
    values = [
        delta["official_like_cost_delta"],
        delta["hpwl_delta"],
        delta["bbox_area_delta"],
        delta["soft_constraint_delta"],
    ]
    return all(value >= -EPS for value in values) and any(value > EPS for value in values)


def quality_gate_status(delta: dict[str, float], hard_feasible: bool, no_op: bool) -> str:
    if no_op:
        return "no_op_report_only"
    if not hard_feasible:
        return "infeasible_after_replay"
    if delta["official_like_cost_delta"] < -EPS and not dominated_by_original(delta, hard_feasible):
        return "archive_candidate"
    if dominated_by_original(delta, hard_feasible):
        return "dominated_by_original"
    return "metric_tradeoff_report_only"


def failure_attribution(status: str, delta: dict[str, float], overlap_violations: int) -> str:
    if status == "archive_candidate":
        return "none"
    if status == "no_op_report_only":
        return "no_op_after_replay"
    if status == "infeasible_after_replay":
        if overlap_violations > 0:
            return "overlap_after_splice"
        return "hard_infeasible_after_replay"
    if delta["soft_constraint_delta"] > EPS:
        return "soft_regression"
    if delta["bbox_area_delta"] > EPS:
        return "bbox_regression"
    if delta["hpwl_delta"] > EPS:
        return "hpwl_regression"
    return "metric_tradeoff"


def summarize_replay_rows(
    rows: list[dict[str, Any]],
    *,
    request_count: int,
    request_path: Path | None = None,
    replay_rows_path: Path | None = None,
    runtime_proxy_ms: float = 0.0,
) -> dict[str, Any]:
    candidates = [row for row in rows if not row.get("is_anchor")]
    exact = [row for row in candidates if row.get("fresh_metric_available")]
    hard = [row for row in exact if row.get("hard_feasible_nonnoop")]
    improving = [row for row in exact if row.get("official_like_cost_improving")]
    winners = [row for row in exact if row.get("quality_gate_status") == "archive_candidate"]
    signatures = {row.get("replayed_signature") for row in exact if row.get("replayed_signature")}
    improving_signatures = {
        row.get("replayed_signature") for row in improving if row.get("replayed_signature")
    }
    route_counts = Counter(str(row.get("actual_locality_class")) for row in exact)
    status_counts = Counter(str(row.get("quality_gate_status")) for row in rows)
    failure_counts = Counter(str(row.get("failure_attribution")) for row in rows)
    case025_outside = [row for row in hard if str(row.get("case_id")) != "25"]
    case025_outside_winners = [row for row in winners if str(row.get("case_id")) != "25"]
    summary = {
        "schema": "step7l_phase2_replay_summary_v1",
        "decision": decide_step7l_completion(rows, winners, improving_signatures),
        "request_path": str(request_path) if request_path is not None else None,
        "replay_rows_path": str(replay_rows_path) if replay_rows_path is not None else None,
        "request_count": request_count,
        "replay_row_count": len(rows),
        "generated_candidate_count": len(candidates),
        "fresh_metric_available_count": len(exact),
        "route_count_by_class": dict(route_counts),
        "global_report_only_count": sum(
            int(bool(row.get("request_global_report_only"))) for row in rows
        ),
        "no_feasible_space_count": status_counts.get("no_feasible_space", 0),
        "timeout_count": 0,
        "full_case_overlap_after_splice_count": failure_counts.get("overlap_after_splice", 0),
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "fresh_official_like_improving_count": len(improving),
        "fresh_quality_gate_pass_count": len(winners),
        "dominated_by_original_count": status_counts.get("dominated_by_original", 0),
        "metric_regressing_count": sum(int(bool(row.get("metric_regressing"))) for row in exact),
        "fresh_hpwl_regression_count": sum(
            int(_float(row.get("hpwl_delta")) > EPS) for row in exact
        ),
        "bbox_regression_count": sum(
            int(_float(row.get("bbox_area_delta")) > EPS) for row in exact
        ),
        "soft_regression_count": sum(
            int(_float(row.get("soft_constraint_delta")) > EPS) for row in exact
        ),
        "case025_outside_solved_count": len(case025_outside),
        "case025_outside_winner_count": len(case025_outside_winners),
        "unique_replayed_signature_count": len(signatures),
        "unique_improving_signature_count": len(improving_signatures),
        "status_counts": dict(status_counts),
        "failure_counts": dict(failure_counts),
        "case_counts": dict(Counter(str(row.get("case_id")) for row in rows)),
        "winner_case_counts": dict(Counter(str(row.get("case_id")) for row in winners)),
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_candidate_ms": runtime_proxy_ms / max(len(candidates), 1),
        "phase3_gnn_gate_open": phase3_gnn_gate_open(winners, improving_signatures),
        "phase4_offline_rl_gate_open": phase4_offline_rl_gate_open(rows, winners, signatures),
        "replay_contract": (
            "request generation is input-only; validation labels used only for replay/evaluation"
        ),
    }
    return summary


def decide_step7l_completion(
    rows: list[dict[str, Any]], winners: list[dict[str, Any]], improving_signatures: set[Any]
) -> str:
    if not rows:
        return "fix_empty_replay"
    if winners and len(improving_signatures) >= 3:
        return "promote_to_step7l_gnn_phase3_gate"
    return "complete_step7l_deterministic_prior_and_defer_gnn_rl"


def phase3_gnn_gate_open(winners: list[dict[str, Any]], improving_signatures: set[Any]) -> bool:
    winner_cases = {str(row.get("case_id")) for row in winners}
    return len(winners) >= 3 and len(improving_signatures) >= 3 and len(winner_cases) >= 2


def phase4_offline_rl_gate_open(
    rows: list[dict[str, Any]], winners: list[dict[str, Any]], signatures: set[Any]
) -> bool:
    winner_cases = {str(row.get("case_id")) for row in winners}
    return (
        len(rows) >= 500
        and len(signatures) >= 100
        and len(winners) >= 10
        and len(winner_cases) >= 3
    )


def failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("failure_attribution"))] += 1
    return {
        "schema": "step7l_phase2_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def decision_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7L Phase 2 Replay / Completion Decision

Decision: `{summary["decision"]}`

## Replay counts

- request_count: {summary["request_count"]}
- generated_candidate_count: {summary["generated_candidate_count"]}
- fresh_metric_available_count: {summary["fresh_metric_available_count"]}
- fresh_hard_feasible_nonnoop_count: {summary["fresh_hard_feasible_nonnoop_count"]}
- fresh_official_like_improving_count: {summary["fresh_official_like_improving_count"]}
- fresh_quality_gate_pass_count: {summary["fresh_quality_gate_pass_count"]}
- unique_replayed_signature_count: {summary["unique_replayed_signature_count"]}
- unique_improving_signature_count: {summary["unique_improving_signature_count"]}
- case025_outside_winner_count: {summary["case025_outside_winner_count"]}
- phase3_gnn_gate_open: {summary["phase3_gnn_gate_open"]}
- phase4_offline_rl_gate_open: {summary["phase4_offline_rl_gate_open"]}

## Interpretation

This is a sidecar replay bridge for Step7L target-window requests. It is not a
contest-runtime change. If Phase 3 / Phase 4 gates are closed, Step7L is complete
for the current deterministic-prior scope and GNN/RL stays deferred until fresh
replay evidence is broad enough.
"""


def replayed_signature(
    case_id: str,
    family: Any,
    block_id: Any,
    positions: list[Box] | None,
) -> str | None:
    if (
        positions is None
        or not isinstance(block_id, int)
        or block_id < 0
        or block_id >= len(positions)
    ):
        return None
    x, y, w, h = positions[block_id]
    return (
        f"case={case_id}|family={family}|block={block_id}|x={x:.3f}|y={y:.3f}|w={w:.3f}|h={h:.3f}"
    )


def _official_like_evaluator(case: FloorSetCase, positions: list[Box]) -> dict[str, Any]:
    return evaluate_positions(case, positions, runtime=1.0)


def _missing_case_row(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "step7l_phase2_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": request.get("case_id"),
        "loader_index": request.get("loader_index"),
        "source_family": request.get("source_family"),
        "block_id": request.get("block_id"),
        "generation_status": "missing_case",
        "quality_gate_status": "missing_case",
        "fresh_metric_available": False,
        "non_original_non_noop": False,
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "official_like_cost_improving": False,
        "dominated_by_original": True,
        "metric_regressing": False,
        "failure_attribution": "missing_case",
    }


def _attempt_json(attempt: ReplayAttempt) -> dict[str, Any]:
    return {
        "status": attempt.status,
        "moved_block_ids": attempt.moved_block_ids,
        "notes": attempt.notes,
        "target_distance": attempt.target_distance,
        "displacement": attempt.displacement,
    }


def _zero_delta() -> dict[str, float]:
    return {
        "official_like_cost_delta": 0.0,
        "hpwl_delta": 0.0,
        "bbox_area_delta": 0.0,
        "soft_constraint_delta": 0.0,
    }


def _moved_blocks(before: list[Box], after: list[Box]) -> list[int]:
    moved = []
    for idx, (src, dst) in enumerate(zip(before, after, strict=False)):
        if any(abs(a - b) > EPS for a, b in zip(src, dst, strict=False)):
            moved.append(idx)
    return moved


def _same_box(a: Box, b: Box) -> bool:
    return all(abs(x - y) <= EPS for x, y in zip(a, b, strict=False))


def _box_center(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _center_distance(box: Box, target: tuple[float, float]) -> float:
    cx, cy = _box_center(box)
    return math.hypot(cx - target[0], cy - target[1])


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
