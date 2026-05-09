"""Step7M objective-guarded replay for corridor requests.

This bridge replays only the exact target windows emitted by Step7M-OAC Phase 1.
Validation labels are consumed only inside the replay/evaluation boundary to
recover the anchor geometry and official-like metrics; no contest runtime or
finalizer code is modified.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import (
    Box,
    load_validation_cases,
    read_jsonl,
    write_jsonl,
)
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.floorset_training_corpus import write_json

EPS = 1e-9
MEANINGFUL_COST_EPS = 1e-7
ReplayEvaluator = Callable[[FloorSetCase, list[Box]], dict[str, Any]]


def replay_corridor_requests(
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
    started = time.perf_counter()
    requests = read_jsonl(requests_path)
    if max_requests is not None:
        requests = requests[:max_requests]
    case_ids = sorted(
        {int(row["loader_index"]) for row in requests if row.get("loader_index") is not None}
    )
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    evaluator = _official_like_evaluator
    baseline_cache: dict[str, tuple[list[Box], dict[str, Any]]] = {}
    replay_rows: list[dict[str, Any]] = []
    for request in requests:
        case = cases.get(int(request.get("loader_index", -1)))
        if case is None:
            replay_rows.append(missing_case_row(request))
            continue
        case_key = str(request.get("case_id"))
        if case_key not in baseline_cache:
            baseline_positions = positions_from_case_targets(case)
            baseline_cache[case_key] = (baseline_positions, evaluator(case, baseline_positions))
        baseline_positions, before_eval = baseline_cache[case_key]
        replay_rows.append(
            replay_corridor_request_row(request, case, baseline_positions, before_eval, evaluator)
        )

    write_jsonl(replay_rows_path, replay_rows)
    summary = summarize_corridor_replay_rows(
        replay_rows,
        request_count=len(requests),
        request_path=requests_path,
        replay_rows_path=replay_rows_path,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
    )
    write_json(summary_path, summary)
    write_json(failures_path, failures_by_case(replay_rows))
    summary_path.with_suffix(".md").write_text(replay_summary_markdown(summary), encoding="utf-8")
    return summary


def replay_corridor_request_row(
    request: dict[str, Any],
    case: FloorSetCase,
    baseline_positions: list[Box],
    before_eval: dict[str, Any],
    evaluator: ReplayEvaluator,
) -> dict[str, Any]:
    base = {
        "schema": "step7m_phase2_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": str(request.get("case_id")),
        "loader_index": request.get("loader_index"),
        "gate_mode": request.get("gate_mode"),
        "source_family": request.get("source_family"),
        "block_id": request.get("block_id"),
        "move_family": request.get("move_family"),
        "request_route_class": request.get("route_class"),
        "accepted_gates": request.get("accepted_gates"),
        "proxy_objective_vector": request.get("proxy_objective_vector"),
        "heatmap_score": request.get("heatmap_score"),
        "replay_scope": "sidecar_exact_single_block_objective_corridor",
        "validation_label_policy": "labels used for replay/evaluation only, not request generation",
        "request_global_report_only": bool(request.get("global_report_only")),
    }
    target = target_box_from_request(request, baseline_positions)
    if target is None:
        return {
            **base,
            "candidate_id": f"{request.get('request_id')}:invalid_target",
            "generation_status": "invalid_request",
            "quality_gate_status": "invalid_request",
            "fresh_metric_available": False,
            "hard_feasible": False,
            "hard_feasible_nonnoop": False,
            "non_original_non_noop": False,
            "official_like_cost_improving": False,
            "dominated_by_original": True,
            "metric_regressing": False,
            "moved_block_ids": [],
            "moved_block_count": 0,
            "actual_objective_vector": None,
            "proxy_actual_signs": None,
            "failure_attribution": "invalid_request",
        }

    block_id = int(request["block_id"])
    if same_box(target, baseline_positions[block_id]):
        return {
            **base,
            "candidate_id": f"{request.get('request_id')}:no_op",
            "generation_status": "no_op_report_only",
            "quality_gate_status": "no_op_report_only",
            "fresh_metric_available": False,
            "hard_feasible": False,
            "hard_feasible_nonnoop": False,
            "non_original_non_noop": False,
            "official_like_cost_improving": False,
            "dominated_by_original": True,
            "metric_regressing": False,
            "moved_block_ids": [],
            "moved_block_count": 0,
            "actual_objective_vector": None,
            "proxy_actual_signs": None,
            "failure_attribution": "no_op_after_exact_replay",
        }
    if overlaps_any(target, baseline_positions, skip=block_id):
        return {
            **base,
            "candidate_id": f"{request.get('request_id')}:overlap",
            "generation_status": "overlap_after_exact_splice",
            "quality_gate_status": "infeasible_after_replay",
            "fresh_metric_available": False,
            "hard_feasible": False,
            "hard_feasible_nonnoop": False,
            "non_original_non_noop": True,
            "official_like_cost_improving": False,
            "dominated_by_original": True,
            "metric_regressing": False,
            "moved_block_ids": [block_id],
            "moved_block_count": 1,
            "actual_objective_vector": None,
            "proxy_actual_signs": None,
            "failure_attribution": "overlap_after_splice",
        }

    after_positions = list(baseline_positions)
    after_positions[block_id] = target
    after_eval = evaluator(case, after_positions)
    legality = summarize_hard_legality(case, after_positions)
    actual = actual_delta(before_eval, after_eval)
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    dominated = dominated_by_original(actual, hard_feasible)
    status = quality_gate_status(actual, hard_feasible)
    proxy_signs = proxy_actual_signs(request.get("proxy_objective_vector"), actual)
    return {
        **base,
        "candidate_id": f"{request.get('request_id')}:step7m:exact_target",
        "generation_status": "realized_exact_target_window",
        "quality_gate_status": status,
        "fresh_metric_available": True,
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible,
        "non_original_non_noop": True,
        "official_like_cost_improving": actual["official_like_cost_delta"] < -EPS,
        "dominated_by_original": dominated,
        "metric_regressing": actual["official_like_cost_delta"] > EPS,
        "moved_block_ids": [block_id],
        "moved_block_count": 1,
        "target_box": target,
        "actual_objective_vector": actual,
        **actual,
        "proxy_actual_signs": proxy_signs,
        "official_before_quality": before_eval["quality"],
        "official_after_quality": after_eval["quality"],
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "failure_attribution": failure_attribution(status, actual, legality.overlap_violations),
        "replayed_signature": replayed_signature(
            str(request.get("case_id")), request.get("gate_mode"), block_id, after_positions
        ),
    }


def target_box_from_request(request: dict[str, Any], positions: list[Box]) -> Box | None:
    block_id = request.get("block_id")
    if not isinstance(block_id, int) or block_id < 0 or block_id >= len(positions):
        return None
    window = request.get("target_window")
    if not isinstance(window, dict):
        return None
    current = positions[block_id]
    x = to_float(window.get("x"), current[0])
    y = to_float(window.get("y"), current[1])
    w = to_float(window.get("w"), current[2])
    h = to_float(window.get("h"), current[3])
    if w <= 0.0 or h <= 0.0:
        return None
    return (x, y, w, h)


def summarize_corridor_replay_rows(
    rows: list[dict[str, Any]],
    *,
    request_count: int,
    request_path: Path | None = None,
    replay_rows_path: Path | None = None,
    runtime_proxy_ms: float = 0.0,
) -> dict[str, Any]:
    exact = [row for row in rows if row.get("fresh_metric_available")]
    hard = [row for row in exact if row.get("hard_feasible_nonnoop")]
    improving = [row for row in hard if row.get("official_like_cost_improving")]
    winners = [row for row in hard if row.get("quality_gate_status") == "archive_candidate"]
    status_counts = Counter(str(row.get("quality_gate_status")) for row in rows)
    failure_counts = Counter(str(row.get("failure_attribution")) for row in rows)
    represented_cases = {str(row.get("case_id")) for row in rows}
    signatures = {row.get("replayed_signature") for row in hard if row.get("replayed_signature")}
    noncase25_winners = [row for row in winners if str(row.get("case_id")) != "25"]
    proxy_precision = proxy_actual_precision(hard)
    regression_rate = sum(int(bool(row.get("metric_regressing"))) for row in hard) / max(
        len(hard), 1
    )
    summary = {
        "schema": "step7m_phase2_replay_summary_v1",
        "decision": decide_step7m_phase2(hard, winners, proxy_precision, regression_rate),
        "request_path": str(request_path) if request_path is not None else None,
        "replay_rows_path": str(replay_rows_path) if replay_rows_path is not None else None,
        "request_count": request_count,
        "replay_row_count": len(rows),
        "fresh_metric_available_count": len(exact),
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "fresh_official_like_improving_count": len(improving),
        "meaningful_official_like_improving_count": sum(
            int(to_float(row.get("official_like_cost_delta")) < -MEANINGFUL_COST_EPS)
            for row in hard
        ),
        "fresh_quality_gate_pass_count": len(winners),
        "actual_metric_regressing_count": sum(
            int(bool(row.get("metric_regressing"))) for row in hard
        ),
        "actual_metric_regression_rate": regression_rate,
        "min_official_like_cost_delta": min(
            (to_float(row.get("official_like_cost_delta")) for row in hard), default=0.0
        ),
        "max_official_like_cost_delta": max(
            (to_float(row.get("official_like_cost_delta")) for row in hard), default=0.0
        ),
        "meaningful_cost_delta_threshold": MEANINGFUL_COST_EPS,
        "actual_hpwl_regression_count": sum(
            int(to_float(row.get("hpwl_delta")) > EPS) for row in hard
        ),
        "actual_bbox_regression_count": sum(
            int(to_float(row.get("bbox_area_delta")) > EPS) for row in hard
        ),
        "actual_soft_regression_count": sum(
            int(to_float(row.get("soft_constraint_delta")) > EPS) for row in hard
        ),
        "actual_all_vector_nonregressing_count": sum(
            int(
                to_float(row.get("hpwl_delta")) <= EPS
                and to_float(row.get("bbox_area_delta")) <= EPS
                and to_float(row.get("soft_constraint_delta")) <= EPS
            )
            for row in hard
        ),
        "case025_request_share": _case_share(rows, "25"),
        "case025_winner_share": _case_share(winners, "25"),
        "case025_outside_winner_count": len(noncase25_winners),
        "represented_case_count": len(represented_cases),
        "unique_replayed_signature_count": len(signatures),
        "status_counts": dict(status_counts),
        "failure_counts": dict(failure_counts),
        "request_count_by_gate_mode": dict(Counter(str(row.get("gate_mode")) for row in rows)),
        "winner_case_counts": dict(Counter(str(row.get("case_id")) for row in winners)),
        "per_case_status_counts": per_case_status_counts(rows),
        "proxy_actual_sign_precision": proxy_precision,
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_candidate_ms": runtime_proxy_ms / max(request_count, 1),
        "phase3_ablation_gate_open": phase3_ablation_gate_open(
            hard, proxy_precision, regression_rate
        ),
        "phase4_multiblock_gate_open": phase4_multiblock_gate_open(hard, winners, regression_rate),
        "gnn_rl_gate_open": gnn_rl_gate_open(hard, winners),
        "replay_contract": (
            "Phase 1 requests are objective-gated; validation labels used only for "
            "exact replay/evaluation"
        ),
    }
    return summary


def decide_step7m_phase2(
    hard: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    proxy_precision: dict[str, Any],
    regression_rate: float,
) -> str:
    if not hard:
        return "fix_step7m_exact_replay_bridge"
    if len(winners) >= 1 and any(str(row.get("case_id")) != "25" for row in winners):
        return "promote_to_archive_integration_review_and_ablation"
    if regression_rate < 1.0 and float(proxy_precision.get("all_component_precision", 0.0)) >= 0.5:
        return "promote_to_step7m_corridor_ablation"
    if regression_rate < 1.0:
        return "promote_to_step7m_corridor_ablation_with_proxy_fix"
    return "close_step7m_single_block_corridor_and_defer_gnn_rl"


def phase3_ablation_gate_open(
    hard: list[dict[str, Any]], proxy_precision: dict[str, Any], regression_rate: float
) -> bool:
    return (
        bool(hard)
        and regression_rate < 1.0
        and float(proxy_precision.get("all_component_precision", 0.0)) >= 0.5
    )


def phase4_multiblock_gate_open(
    hard: list[dict[str, Any]], winners: list[dict[str, Any]], regression_rate: float
) -> bool:
    return bool(hard) and regression_rate < 1.0 and not winners


def gnn_rl_gate_open(hard: list[dict[str, Any]], winners: list[dict[str, Any]]) -> bool:
    winner_cases = {str(row.get("case_id")) for row in winners}
    return len(hard) >= 1000 and len(winners) >= 10 and len(winner_cases) >= 4


def proxy_actual_precision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_total = 0
    component_match = 0
    all_component_match = 0
    by_component: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        signs = row.get("proxy_actual_signs")
        if not isinstance(signs, dict):
            continue
        row_matches = []
        for component in ("hpwl", "bbox", "soft"):
            match = bool(signs.get(f"{component}_nonregression_match"))
            row_matches.append(match)
            by_component[component].append(match)
            component_total += 1
            component_match += int(match)
        all_component_match += int(all(row_matches))
    return {
        "component_match_count": component_match,
        "component_total": component_total,
        "component_precision": component_match / max(component_total, 1),
        "all_component_match_count": all_component_match,
        "all_component_total": len(rows),
        "all_component_precision": all_component_match / max(len(rows), 1),
        "by_component": {
            component: sum(matches) / max(len(matches), 1)
            for component, matches in sorted(by_component.items())
        },
    }


def proxy_actual_signs(proxy: Any, actual: dict[str, float]) -> dict[str, bool | float | None]:
    proxy = proxy if isinstance(proxy, dict) else {}
    proxy_soft = soft_sum(proxy)
    actual_soft = actual["soft_constraint_delta"]
    return {
        "hpwl_proxy_delta": to_float(proxy.get("hpwl_delta_proxy")),
        "hpwl_actual_delta": actual["hpwl_delta"],
        "hpwl_proxy_nonregress": to_float(proxy.get("hpwl_delta_proxy")) <= EPS,
        "hpwl_actual_nonregress": actual["hpwl_delta"] <= EPS,
        "hpwl_nonregression_match": (to_float(proxy.get("hpwl_delta_proxy")) <= EPS)
        == (actual["hpwl_delta"] <= EPS),
        "bbox_proxy_delta": to_float(proxy.get("bbox_area_delta_proxy")),
        "bbox_actual_delta": actual["bbox_area_delta"],
        "bbox_proxy_nonregress": to_float(proxy.get("bbox_area_delta_proxy")) <= EPS,
        "bbox_actual_nonregress": actual["bbox_area_delta"] <= EPS,
        "bbox_nonregression_match": (to_float(proxy.get("bbox_area_delta_proxy")) <= EPS)
        == (actual["bbox_area_delta"] <= EPS),
        "soft_proxy_delta": proxy_soft,
        "soft_actual_delta": actual_soft,
        "soft_proxy_nonregress": proxy_soft <= EPS,
        "soft_actual_nonregress": actual_soft <= EPS,
        "soft_nonregression_match": (proxy_soft <= EPS) == (actual_soft <= EPS),
    }


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


def quality_gate_status(delta: dict[str, float], hard_feasible: bool) -> str:
    if not hard_feasible:
        return "infeasible_after_replay"
    if delta["official_like_cost_delta"] < -MEANINGFUL_COST_EPS and not dominated_by_original(
        delta, hard_feasible
    ):
        return "archive_candidate"
    if dominated_by_original(delta, hard_feasible):
        return "dominated_by_original"
    return "metric_tradeoff_report_only"


def failure_attribution(status: str, delta: dict[str, float], overlap_violations: int) -> str:
    if status == "archive_candidate":
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


def failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("failure_attribution"))] += 1
    return {
        "schema": "step7m_phase2_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def per_case_status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("quality_gate_status"))] += 1
    return {case_id: dict(counter) for case_id, counter in sorted(by_case.items())}


def replay_summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7M Phase 2 Objective-Guarded Replay

Decision: `{summary["decision"]}`

- request_count: {summary["request_count"]}
- fresh_hard_feasible_nonnoop_count: {summary["fresh_hard_feasible_nonnoop_count"]}
- fresh_official_like_improving_count: {summary["fresh_official_like_improving_count"]}
- fresh_quality_gate_pass_count: {summary["fresh_quality_gate_pass_count"]}
- actual_metric_regressing_count: {summary["actual_metric_regressing_count"]}
- actual_metric_regression_rate: {summary["actual_metric_regression_rate"]:.3f}
- actual_hpwl_regression_count: {summary["actual_hpwl_regression_count"]}
- actual_bbox_regression_count: {summary["actual_bbox_regression_count"]}
- actual_soft_regression_count: {summary["actual_soft_regression_count"]}
- actual_all_vector_nonregressing_count: {summary["actual_all_vector_nonregressing_count"]}
- represented_case_count: {summary["represented_case_count"]}
- unique_replayed_signature_count: {summary["unique_replayed_signature_count"]}
- proxy_actual_sign_precision: {summary["proxy_actual_sign_precision"]}
- phase3_ablation_gate_open: {summary["phase3_ablation_gate_open"]}
- phase4_multiblock_gate_open: {summary["phase4_multiblock_gate_open"]}
- gnn_rl_gate_open: {summary["gnn_rl_gate_open"]}
"""


def missing_case_row(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "step7m_phase2_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": str(request.get("case_id")),
        "loader_index": request.get("loader_index"),
        "gate_mode": request.get("gate_mode"),
        "source_family": request.get("source_family"),
        "block_id": request.get("block_id"),
        "generation_status": "missing_case",
        "quality_gate_status": "missing_case",
        "fresh_metric_available": False,
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "non_original_non_noop": False,
        "official_like_cost_improving": False,
        "dominated_by_original": True,
        "metric_regressing": False,
        "failure_attribution": "missing_case",
    }


def overlaps_any(candidate: Box, positions: list[Box], *, skip: int) -> bool:
    return any(
        index != skip and overlap_area(candidate, other) > EPS
        for index, other in enumerate(positions)
    )


def overlap_area(a: Box, b: Box) -> float:
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def same_box(a: Box, b: Box) -> bool:
    return all(abs(x - y) <= EPS for x, y in zip(a, b, strict=False))


def replayed_signature(case_id: str, gate_mode: Any, block_id: int, positions: list[Box]) -> str:
    x, y, w, h = positions[block_id]
    return (
        f"case={case_id}|gate={gate_mode}|block={block_id}|x={x:.3f}|y={y:.3f}|w={w:.3f}|h={h:.3f}"
    )


def soft_sum(vector: dict[str, Any]) -> float:
    return (
        to_float(vector.get("boundary_delta_proxy"))
        + to_float(vector.get("group_delta_proxy"))
        + to_float(vector.get("mib_delta_proxy"))
    )


def _case_share(rows: list[dict[str, Any]], case_id: str) -> float:
    if not rows:
        return 0.0
    return sum(int(str(row.get("case_id")) == case_id) for row in rows) / len(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _official_like_evaluator(case: FloorSetCase, positions: list[Box]) -> dict[str, Any]:
    return evaluate_positions(case, positions, runtime=1.0)
