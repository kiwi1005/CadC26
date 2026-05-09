"""Step7M Phase 4 deterministic paired/block-shift corridors."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from itertools import combinations
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


def generate_multiblock_requests(
    phase1_requests_path: Path,
    opportunity_atlas_path: Path,
    out_path: Path,
    summary_path: Path,
    *,
    max_pairs_per_case: int = 24,
    include_soft_budgeted: bool = False,
) -> dict[str, Any]:
    requests = read_jsonl(phase1_requests_path)
    atlas = load_atlas_current_boxes(opportunity_atlas_path)
    candidates = [
        row
        for row in requests
        if is_strict_candidate(row, include_soft_budgeted=include_soft_budgeted)
    ]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_case[str(row.get("case_id"))].append(row)

    out_rows: list[dict[str, Any]] = []
    for _case_id, rows in sorted(by_case.items(), key=lambda item: int(item[0])):
        case_rows: list[dict[str, Any]] = []
        for left, right in combinations(sorted(rows, key=request_sort_key), 2):
            if left.get("block_id") == right.get("block_id"):
                continue
            pair = make_pair_request(left, right, atlas)
            if pair is None:
                continue
            case_rows.append(pair)
        case_rows.sort(key=pair_sort_key)
        out_rows.extend(case_rows[:max_pairs_per_case])

    write_jsonl(out_path, out_rows)
    summary = summarize_multiblock_requests(
        out_rows,
        source_request_count=len(requests),
        strict_candidate_count=len(candidates),
        out_path=out_path,
        include_soft_budgeted=include_soft_budgeted,
    )
    write_json(summary_path, summary)
    summary_path.with_suffix(".md").write_text(
        multiblock_request_markdown(summary), encoding="utf-8"
    )
    return summary


def load_atlas_current_boxes(path: Path) -> dict[tuple[str, int], Box]:
    rows = read_jsonl(path)
    boxes: dict[tuple[str, int], Box] = {}
    for row in rows:
        box = row.get("current_box")
        block_id = row.get("block_id")
        if isinstance(block_id, int) and isinstance(box, list | tuple) and len(box) == 4:
            boxes[(str(row.get("case_id")), block_id)] = tuple(float(value) for value in box)  # type: ignore[assignment]
    return boxes


def is_strict_candidate(row: dict[str, Any], *, include_soft_budgeted: bool) -> bool:
    if row.get("source_family") != "micro_axis_corridor":
        return False
    if row.get("gate_mode") == "soft_repair_budgeted" and not include_soft_budgeted:
        return False
    proxy = proxy_vector(row)
    return (
        proxy_float(proxy, "hpwl_delta_proxy") <= EPS
        and proxy_float(proxy, "bbox_area_delta_proxy") <= EPS
        and proxy_soft(proxy) <= EPS
        and proxy_float(proxy, "overlap_risk_proxy") <= EPS
    )


def make_pair_request(
    left: dict[str, Any], right: dict[str, Any], atlas: dict[tuple[str, int], Box]
) -> dict[str, Any] | None:
    case_id = str(left.get("case_id"))
    if case_id != str(right.get("case_id")):
        return None
    left_block = left.get("block_id")
    right_block = right.get("block_id")
    if not isinstance(left_block, int) or not isinstance(right_block, int):
        return None
    left_current = atlas.get((case_id, left_block))
    right_current = atlas.get((case_id, right_block))
    if left_current is None or right_current is None:
        return None
    left_target = target_box(left)
    right_target = target_box(right)
    if left_target is None or right_target is None:
        return None
    if overlap_area(left_target, right_target) > EPS:
        return None
    left_axis = dominant_axis(left_current, left_target)
    right_axis = dominant_axis(right_current, right_target)
    if left_axis == "none" or left_axis != right_axis:
        return None
    aggregate = aggregate_proxy([left, right])
    if not strict_aggregate_ok(aggregate):
        return None
    request_id = (
        f"step7m_phase4_pair_case{case_id}_{left_axis}_"
        f"b{left_block}_b{right_block}_"
        f"{short_coord(left_target)}_{short_coord(right_target)}"
    )
    return {
        "schema": "step7m_phase4_multiblock_request_v1",
        "case_id": case_id,
        "loader_index": left.get("loader_index"),
        "request_id": request_id,
        "source_family": "paired_micro_axis_strict",
        "move_family": f"paired_{left_axis}_block_shift_strict",
        "member_request_ids": [left.get("request_id"), right.get("request_id")],
        "moved_blocks": [
            moved_block_payload(left_block, left_current, left_target),
            moved_block_payload(right_block, right_current, right_target),
        ],
        "proxy_objective_vector": aggregate,
        "accepted_gates": {"overlap": True, "hpwl": True, "bbox": True, "soft": True},
        "global_report_only": False,
        "provenance": {
            "source": "step7m_phase4_deterministic_multiblock_corridor",
            "label_policy": "uses Phase1 proxy-safe requests and Phase0 anchor boxes only; "
            "no model training",
            "gnn_rl_gate": "closed",
        },
    }


def replay_multiblock_requests(
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
    baseline_cache: dict[str, tuple[list[Box], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for request in requests:
        case = cases.get(int(request.get("loader_index", -1)))
        if case is None:
            rows.append(missing_case_row(request))
            continue
        case_key = str(request.get("case_id"))
        if case_key not in baseline_cache:
            baseline = positions_from_case_targets(case)
            baseline_cache[case_key] = (baseline, evaluate_positions(case, baseline, runtime=1.0))
        baseline, before_eval = baseline_cache[case_key]
        rows.append(replay_multiblock_row(request, case, baseline, before_eval))

    write_jsonl(replay_rows_path, rows)
    summary = summarize_multiblock_replay(
        rows,
        request_count=len(requests),
        requests_path=requests_path,
        replay_rows_path=replay_rows_path,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
    )
    write_json(summary_path, summary)
    write_json(failures_path, failures_by_case(rows))
    summary_path.with_suffix(".md").write_text(
        multiblock_replay_markdown(summary), encoding="utf-8"
    )
    return summary


def replay_multiblock_row(
    request: dict[str, Any],
    case: FloorSetCase,
    baseline: list[Box],
    before_eval: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "schema": "step7m_phase4_multiblock_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": str(request.get("case_id")),
        "loader_index": request.get("loader_index"),
        "source_family": request.get("source_family"),
        "move_family": request.get("move_family"),
        "proxy_objective_vector": request.get("proxy_objective_vector"),
        "member_request_ids": request.get("member_request_ids"),
        "replay_scope": "sidecar_exact_multiblock_corridor",
    }
    positions = list(baseline)
    moved_blocks = request.get("moved_blocks")
    if not isinstance(moved_blocks, list) or not moved_blocks:
        return unrealized_row(base, "invalid_request", [], False)
    block_ids: list[int] = []
    for item in moved_blocks:
        if not isinstance(item, dict) or not isinstance(item.get("block_id"), int):
            return unrealized_row(base, "invalid_request", block_ids, False)
        block_id = int(item["block_id"])
        box = item.get("target_box")
        if block_id < 0 or block_id >= len(positions) or not is_box(box):
            return unrealized_row(base, "invalid_request", block_ids, False)
        assert isinstance(box, list | tuple)
        positions[block_id] = tuple(float(value) for value in box)  # type: ignore[assignment]
        block_ids.append(block_id)
    if has_overlap(positions):
        return unrealized_row(base, "overlap_after_splice", block_ids, True)
    after_eval = evaluate_positions(case, positions, runtime=1.0)
    legality = summarize_hard_legality(case, positions)
    actual = actual_delta(before_eval, after_eval)
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    status = quality_gate_status(actual, hard_feasible)
    return {
        **base,
        "candidate_id": f"{request.get('request_id')}:step7m:exact_multiblock",
        "generation_status": "realized_exact_multiblock_request",
        "quality_gate_status": status,
        "fresh_metric_available": True,
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible,
        "non_original_non_noop": True,
        "official_like_cost_improving": actual["official_like_cost_delta"] < -EPS,
        "meaningful_official_like_improving": actual["official_like_cost_delta"]
        < -MEANINGFUL_COST_EPS,
        "metric_regressing": actual["official_like_cost_delta"] > EPS,
        "moved_block_ids": block_ids,
        "moved_block_count": len(block_ids),
        "actual_objective_vector": actual,
        **actual,
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "failure_attribution": failure_attribution(status, actual, legality.overlap_violations),
        "official_before_quality": before_eval["quality"],
        "official_after_quality": after_eval["quality"],
        "replayed_signature": replayed_signature(str(request.get("case_id")), block_ids, positions),
    }


def summarize_multiblock_requests(
    rows: list[dict[str, Any]],
    *,
    source_request_count: int,
    strict_candidate_count: int,
    out_path: Path,
    include_soft_budgeted: bool,
) -> dict[str, Any]:
    cases = {str(row.get("case_id")) for row in rows}
    return {
        "schema": "step7m_phase4_multiblock_request_summary_v1",
        "decision": "promote_to_multiblock_replay" if rows else "fix_empty_multiblock_requests",
        "source_request_count": source_request_count,
        "strict_candidate_count": strict_candidate_count,
        "request_count": len(rows),
        "represented_case_count": len(cases),
        "unique_signature_count": len({row.get("request_id") for row in rows}),
        "include_soft_budgeted": include_soft_budgeted,
        "request_count_by_case": dict(Counter(str(row.get("case_id")) for row in rows)),
        "request_count_by_move_family": dict(Counter(str(row.get("move_family")) for row in rows)),
        "out_path": str(out_path),
        "gnn_rl_gate_open": False,
    }


def summarize_multiblock_replay(
    rows: list[dict[str, Any]],
    *,
    request_count: int,
    requests_path: Path,
    replay_rows_path: Path,
    runtime_proxy_ms: float,
) -> dict[str, Any]:
    hard = [row for row in rows if row.get("hard_feasible_nonnoop")]
    meaningful = [row for row in hard if row.get("meaningful_official_like_improving")]
    winners = [row for row in hard if row.get("quality_gate_status") == "archive_candidate"]
    regressions = sum(int(bool(row.get("metric_regressing"))) for row in hard)
    regression_rate = regressions / max(len(hard), 1)
    decision = "complete_step7m_deterministic_multiblock_and_defer_gnn_rl"
    if meaningful:
        decision = "promote_to_archive_integration_review"
    return {
        "schema": "step7m_phase4_multiblock_replay_summary_v1",
        "decision": decision,
        "request_path": str(requests_path),
        "replay_rows_path": str(replay_rows_path),
        "request_count": request_count,
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "fresh_quality_gate_pass_count": len(winners),
        "meaningful_official_like_improving_count": len(meaningful),
        "actual_metric_regressing_count": regressions,
        "actual_metric_regression_rate": regression_rate,
        "actual_hpwl_regression_count": sum(int(hpwl_delta(row) > EPS) for row in hard),
        "actual_bbox_regression_count": sum(int(bbox_delta(row) > EPS) for row in hard),
        "actual_soft_regression_count": sum(int(soft_delta(row) > EPS) for row in hard),
        "actual_all_vector_nonregressing_count": sum(
            int(hpwl_delta(row) <= EPS and bbox_delta(row) <= EPS and soft_delta(row) <= EPS)
            for row in hard
        ),
        "represented_case_count": len({str(row.get("case_id")) for row in rows}),
        "status_counts": dict(Counter(str(row.get("quality_gate_status")) for row in rows)),
        "failure_counts": dict(Counter(str(row.get("failure_attribution")) for row in rows)),
        "winner_case_counts": dict(Counter(str(row.get("case_id")) for row in winners)),
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_candidate_ms": runtime_proxy_ms / max(request_count, 1),
        "gnn_rl_gate_open": False,
    }


def moved_block_payload(block_id: int, current: Box, target: Box) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "current_box": current,
        "target_box": target,
        "dx": target[0] - current[0],
        "dy": target[1] - current[1],
    }


def dominant_axis(current: Box, target: Box) -> str:
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) <= EPS and abs(dy) <= EPS:
        return "none"
    if abs(dx) >= abs(dy):
        return "x_pos" if dx > 0 else "x_neg"
    return "y_pos" if dy > 0 else "y_neg"


def aggregate_proxy(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "hpwl_delta_proxy": sum(proxy_float(proxy_vector(row), "hpwl_delta_proxy") for row in rows),
        "bbox_area_delta_proxy": sum(
            proxy_float(proxy_vector(row), "bbox_area_delta_proxy") for row in rows
        ),
        "boundary_delta_proxy": sum(
            proxy_float(proxy_vector(row), "boundary_delta_proxy") for row in rows
        ),
        "group_delta_proxy": sum(
            proxy_float(proxy_vector(row), "group_delta_proxy") for row in rows
        ),
        "mib_delta_proxy": sum(proxy_float(proxy_vector(row), "mib_delta_proxy") for row in rows),
        "overlap_risk_proxy": 0.0,
    }


def strict_aggregate_ok(vector: dict[str, float]) -> bool:
    return (
        vector["hpwl_delta_proxy"] <= EPS
        and vector["bbox_area_delta_proxy"] <= EPS
        and vector["boundary_delta_proxy"] + vector["group_delta_proxy"] + vector["mib_delta_proxy"]
        <= EPS
    )


def target_box(row: dict[str, Any]) -> Box | None:
    window = row.get("target_window")
    if not isinstance(window, dict):
        return None
    try:
        return (
            float(window["x"]),
            float(window["y"]),
            float(window["w"]),
            float(window["h"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def target_window_sort_value(row: dict[str, Any]) -> tuple[float, float]:
    box = target_box(row)
    return (0.0, 0.0) if box is None else (box[0], box[1])


def request_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    proxy = proxy_vector(row)
    return (
        str(row.get("case_id")),
        int(row.get("block_id", -1)),
        proxy_float(proxy, "hpwl_delta_proxy"),
        proxy_soft(proxy),
        target_window_sort_value(row),
        str(row.get("request_id")),
    )


def pair_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    proxy = proxy_vector(row)
    return (
        proxy_float(proxy, "hpwl_delta_proxy"),
        proxy_soft(proxy),
        str(row.get("case_id")),
        str(row.get("request_id")),
    )


def actual_delta(before_eval: dict[str, Any], after_eval: dict[str, Any]) -> dict[str, float]:
    before = before_eval["quality"]
    after = after_eval["quality"]
    return {
        "official_like_cost_delta": to_float(after.get("cost")) - to_float(before.get("cost")),
        "hpwl_delta": to_float(after.get("HPWLgap")) - to_float(before.get("HPWLgap")),
        "bbox_area_delta": to_float(after.get("Areagap_bbox"))
        - to_float(before.get("Areagap_bbox")),
        "soft_constraint_delta": to_float(after.get("Violationsrelative"))
        - to_float(before.get("Violationsrelative")),
    }


def quality_gate_status(delta: dict[str, float], hard_feasible: bool) -> str:
    if not hard_feasible:
        return "infeasible_after_replay"
    if delta["official_like_cost_delta"] < -MEANINGFUL_COST_EPS:
        return "archive_candidate"
    if delta["official_like_cost_delta"] > EPS:
        return "dominated_by_original"
    return "metric_tradeoff_report_only"


def failure_attribution(status: str, delta: dict[str, float], overlap_violations: int) -> str:
    if status == "archive_candidate":
        return "none"
    if status == "infeasible_after_replay":
        return "overlap_after_splice" if overlap_violations else "hard_infeasible_after_replay"
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
        "schema": "step7m_phase4_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def unrealized_row(
    base: dict[str, Any], status: str, block_ids: list[int], nonnoop: bool
) -> dict[str, Any]:
    return {
        **base,
        "candidate_id": f"{base.get('request_id')}:{status}",
        "generation_status": status,
        "quality_gate_status": status,
        "fresh_metric_available": False,
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "non_original_non_noop": nonnoop,
        "official_like_cost_improving": False,
        "meaningful_official_like_improving": False,
        "metric_regressing": False,
        "moved_block_ids": block_ids,
        "moved_block_count": len(block_ids),
        "failure_attribution": status,
    }


def multiblock_request_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7M Phase 4 Multiblock Requests

Decision: `{summary["decision"]}`

- source_request_count: {summary["source_request_count"]}
- strict_candidate_count: {summary["strict_candidate_count"]}
- request_count: {summary["request_count"]}
- represented_case_count: {summary["represented_case_count"]}
- include_soft_budgeted: {summary["include_soft_budgeted"]}
- gnn_rl_gate_open: {summary["gnn_rl_gate_open"]}
"""


def multiblock_replay_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7M Phase 4 Multiblock Replay

Decision: `{summary["decision"]}`

- request_count: {summary["request_count"]}
- fresh_hard_feasible_nonnoop_count: {summary["fresh_hard_feasible_nonnoop_count"]}
- fresh_quality_gate_pass_count: {summary["fresh_quality_gate_pass_count"]}
- meaningful_official_like_improving_count: {summary["meaningful_official_like_improving_count"]}
- actual_metric_regressing_count: {summary["actual_metric_regressing_count"]}
- actual_metric_regression_rate: {summary["actual_metric_regression_rate"]:.3f}
- actual_hpwl_regression_count: {summary["actual_hpwl_regression_count"]}
- actual_bbox_regression_count: {summary["actual_bbox_regression_count"]}
- actual_soft_regression_count: {summary["actual_soft_regression_count"]}
- actual_all_vector_nonregressing_count: {summary["actual_all_vector_nonregressing_count"]}
- represented_case_count: {summary["represented_case_count"]}
- gnn_rl_gate_open: {summary["gnn_rl_gate_open"]}
"""


def has_overlap(positions: list[Box]) -> bool:
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if overlap_area(left, right) > EPS:
                return True
    return False


def overlap_area(a: Box, b: Box) -> float:
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def is_box(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 4


def proxy_vector(row: dict[str, Any]) -> dict[str, Any]:
    proxy = row.get("proxy_objective_vector")
    return proxy if isinstance(proxy, dict) else {}


def proxy_float(proxy: dict[str, Any], key: str) -> float:
    return to_float(proxy.get(key))


def proxy_soft(proxy: dict[str, Any]) -> float:
    return (
        proxy_float(proxy, "boundary_delta_proxy")
        + proxy_float(proxy, "group_delta_proxy")
        + proxy_float(proxy, "mib_delta_proxy")
    )


def hpwl_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("hpwl_delta"))


def bbox_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("bbox_area_delta"))


def soft_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("soft_constraint_delta"))


def replayed_signature(case_id: str, block_ids: list[int], positions: list[Box]) -> str:
    parts = []
    for block_id in sorted(block_ids):
        x, y, w, h = positions[block_id]
        parts.append(f"b{block_id}:x{x:.3f}:y{y:.3f}:w{w:.3f}:h{h:.3f}")
    return f"case={case_id}|" + "|".join(parts)


def short_coord(box: Box) -> str:
    return f"x{box[0]:.3f}_y{box[1]:.3f}"


def missing_case_row(request: dict[str, Any]) -> dict[str, Any]:
    return unrealized_row(
        {
            "schema": "step7m_phase4_multiblock_replay_row_v1",
            "request_id": request.get("request_id"),
            "case_id": str(request.get("case_id")),
            "loader_index": request.get("loader_index"),
            "source_family": request.get("source_family"),
            "move_family": request.get("move_family"),
        },
        "missing_case",
        [],
        False,
    )


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
