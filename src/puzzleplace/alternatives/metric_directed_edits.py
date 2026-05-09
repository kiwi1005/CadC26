from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.alternatives.real_placement_edits import (
    bbox_area,
    changed_blocks,
    frame_from_baseline,
    free_space_fit_ratio,
    metric_delta_report,
    minimum_region_slack,
    movable_blocks,
    official_like_hard_summary,
    placements_from_case,
    route_appropriate_real_repair_proxy,
    soft_delta_components,
    touched_region_count,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.research.move_library import hpwl_proxy
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, _bbox_from_placements

STRATEGIES = (
    "original_layout",
    "hpwl_directed_local_nudge",
    "bbox_shrink_nudge",
    "slack_fit_insertion",
    "balanced_swap",
    "regional_metric_directed_shift",
    "macro_closure_metric_move",
    "legacy_step7g_global_report",
)

EXPECTED_CLASS_BY_STRATEGY = {
    "original_layout": "local",
    "hpwl_directed_local_nudge": "local",
    "bbox_shrink_nudge": "local",
    "slack_fit_insertion": "local",
    "balanced_swap": "local",
    "regional_metric_directed_shift": "regional",
    "macro_closure_metric_move": "macro",
    "legacy_step7g_global_report": "global",
}


@dataclass(frozen=True)
class MetricDirectedEditCandidate:
    case_id: int
    candidate_id: str
    strategy: str
    descriptor_locality_class: str
    descriptor_repair_mode: str
    case: FloorSetCase
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    changed_blocks: tuple[int, ...]
    macro_closure_blocks: tuple[int, ...]
    predicted_macro_closure_size: int
    construction_status: str
    construction_notes: tuple[str, ...]
    construction_ms: float
    internal_trial_count: int
    internal_feasible_trial_count: int
    internal_best_official_like_cost_delta: float


@dataclass(frozen=True)
class _CaseMetricContext:
    case: FloorSetCase
    baseline: dict[int, Placement]
    frame: PuzzleFrame
    baseline_quality: dict[str, float]
    baseline_hpwl: float
    baseline_bbox_area: float
    baseline_soft_total: int


@dataclass(frozen=True)
class _Trial:
    edited: dict[int, Placement]
    changed: set[int]
    macro_closure: set[int]
    note: str
    proxy_key: tuple[float, float, float, float]
    official_like_cost_delta: float | None = None


@dataclass(frozen=True)
class _BuildResult:
    edited: dict[int, Placement]
    changed: set[int]
    macro_closure: set[int]
    status: str
    notes: tuple[str, ...]
    trial_count: int
    feasible_trial_count: int
    best_cost_delta: float


def build_metric_directed_edit_candidates(
    case_ids: list[int],
    cases_by_id: dict[int, FloorSetCase],
) -> list[MetricDirectedEditCandidate]:
    out: list[MetricDirectedEditCandidate] = []
    for case_id in case_ids:
        case = cases_by_id[case_id]
        baseline = placements_from_case(case)
        ctx = _case_context(case, baseline)
        for strategy in STRATEGIES:
            t0 = time.perf_counter()
            result = _build_strategy_candidate(ctx, strategy)
            expected = EXPECTED_CLASS_BY_STRATEGY[strategy]
            out.append(
                MetricDirectedEditCandidate(
                    case_id=case_id,
                    candidate_id=f"case{case_id:03d}:{strategy}",
                    strategy=strategy,
                    descriptor_locality_class=expected,
                    descriptor_repair_mode=_route_for_expected(expected),
                    case=case,
                    baseline=baseline,
                    edited=result.edited,
                    frame=ctx.frame,
                    changed_blocks=tuple(sorted(changed_blocks(baseline, result.edited))),
                    macro_closure_blocks=tuple(sorted(result.macro_closure)),
                    predicted_macro_closure_size=_predicted_macro_size(
                        strategy, case.block_count, result
                    ),
                    construction_status=result.status,
                    construction_notes=result.notes,
                    construction_ms=(time.perf_counter() - t0) * 1000.0,
                    internal_trial_count=result.trial_count,
                    internal_feasible_trial_count=result.feasible_trial_count,
                    internal_best_official_like_cost_delta=result.best_cost_delta,
                )
            )
    return out


def evaluate_metric_directed_candidate(candidate: MetricDirectedEditCandidate) -> dict[str, Any]:
    hard_before = official_like_hard_summary(candidate.case, candidate.edited, candidate.frame)
    changed = set(candidate.changed_blocks)
    regions = touched_region_count(candidate.edited, changed, candidate.frame)
    raw_fit = free_space_fit_ratio(candidate.edited, changed, candidate.frame)
    fit_ratio = min(raw_fit, 0.95) if hard_before["official_like_hard_feasible"] else raw_fit
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.case.block_count,
        changed_block_count=len(changed),
        touched_region_count=regions,
        macro_closure_size=candidate.predicted_macro_closure_size,
        min_region_slack=minimum_region_slack(candidate.edited, candidate.frame),
        free_space_fit_ratio=fit_ratio,
        hard_summary={"hard_feasible": hard_before["official_like_hard_feasible"]},
    )
    actual_class = str(prediction["predicted_locality_class"])
    repair = route_appropriate_real_repair_proxy(candidate, actual_class, hard_before)  # type: ignore[arg-type]
    metrics = metric_delta_report(candidate.case, candidate.baseline, repair["after_route"])
    row: dict[str, Any] = {
        "case_id": candidate.case_id,
        "candidate_id": candidate.candidate_id,
        "strategy": candidate.strategy,
        "family": candidate.strategy,
        "descriptor_locality_class": candidate.descriptor_locality_class,
        "descriptor_repair_mode": candidate.descriptor_repair_mode,
        "actual_locality_class": actual_class,
        "actual_repair_mode": prediction["predicted_repair_mode"],
        "route_lane": repair["route_lane"],
        "report_only": repair["report_only"],
        "construction_status": candidate.construction_status,
        "construction_notes": list(candidate.construction_notes),
        "real_block_ids_changed": list(candidate.changed_blocks),
        "macro_closure_block_ids": list(candidate.macro_closure_blocks),
        "changed_block_count": len(changed),
        "changed_block_fraction": len(changed) / max(candidate.case.block_count, 1),
        "affected_region_count": regions,
        "macro_closure_size": candidate.predicted_macro_closure_size,
        "free_space_fit_ratio": fit_ratio,
        "internal_trial_count": candidate.internal_trial_count,
        "internal_feasible_trial_count": candidate.internal_feasible_trial_count,
        "internal_best_official_like_cost_delta": candidate.internal_best_official_like_cost_delta,
        **{f"before_repair_{key}": value for key, value in hard_before.items()},
        **{f"after_route_{key}": value for key, value in repair["after_hard"].items()},
        **metrics,
    }
    row["runtime_proxy_ms"] = candidate.construction_ms + float(metrics["metric_eval_ms"])
    row["no_op"] = len(changed) == 0 and candidate.strategy != "original_layout"
    row["official_like_cost_improving"] = _official_improving(row)
    row["safe_improvement"] = is_safe_improvement(row)
    row["failure_attribution"] = failure_attribution(row)
    return row


def evaluate_metric_directed_edits(
    candidates: list[MetricDirectedEditCandidate],
) -> list[dict[str, Any]]:
    return [evaluate_metric_directed_candidate(candidate) for candidate in candidates]


def route_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row["actual_locality_class"]) for row in rows)
    local_rows = [row for row in rows if row["actual_repair_mode"] == "bounded_repair_pareto"]
    invalid_local = [
        row for row in local_rows if not row["before_repair_official_like_hard_feasible"]
    ]
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    stable = 0
    for row in rows:
        descriptor = str(row["descriptor_locality_class"])
        actual = str(row["actual_locality_class"])
        matrix[descriptor][actual] += 1
        stable += int(descriptor == actual)
    return {
        "route_count_by_class": dict(class_counts),
        "non_global_candidate_rate": _rate(len(rows) - class_counts["global"], len(rows)),
        "invalid_local_attempt_rate": _rate(len(invalid_local), len(local_rows)),
        "descriptor_to_real_route_stability": _rate(stable, len(rows)),
        "descriptor_class_vs_actual_route_confusion": {
            key: dict(value) for key, value in sorted(matrix.items())
        },
        "global_report_only_count": sum(int(bool(row["report_only"])) for row in rows),
        "collapsed_to_global": [
            _compact_identity(row)
            for row in rows
            if row["descriptor_locality_class"] != "global"
            and row["actual_locality_class"] == "global"
        ],
    }


def feasibility_report(
    rows: list[dict[str, Any]],
    *,
    real_case_count: int,
    real_a_feasibility: dict[str, Any],
) -> dict[str, Any]:
    feasible = [row for row in rows if row["after_route_official_like_hard_feasible"]]
    improving = [row for row in feasible if row["safe_improvement"]]
    cost_improving = [row for row in feasible if row["official_like_cost_improving"]]
    regional_macro = [row for row in rows if row["actual_locality_class"] in {"regional", "macro"}]
    regional_macro_non_noop = [row for row in regional_macro if not row["no_op"]]
    regional_macro_feasible = [
        row for row in regional_macro if row["after_route_official_like_hard_feasible"]
    ]
    per_case_runtime: dict[str, float] = defaultdict(float)
    for row in rows:
        per_case_runtime[str(row["case_id"])] += float(row["runtime_proxy_ms"])
    counts = _strategy_counts(rows)
    poor = [row for row in rows if str(row["failure_attribution"]).startswith("poor_targeting")]
    legalizer = [
        row for row in rows if str(row["failure_attribution"]).startswith("legalizer_limited")
    ]
    return {
        "real_case_count": real_case_count,
        "candidate_count_by_strategy": counts["candidate"],
        "feasible_count_by_strategy": counts["feasible"],
        "improving_count_by_strategy": counts["improving"],
        "official_like_cost_improving_count": len(cost_improving),
        "improvement_density": _rate(len(improving), len(feasible)),
        "official_like_cost_improvement_density": _rate(len(cost_improving), len(feasible)),
        "route_count_by_class": dict(Counter(str(row["actual_locality_class"]) for row in rows)),
        "non_global_candidate_rate": _rate(
            sum(int(row["actual_locality_class"] != "global") for row in rows), len(rows)
        ),
        "invalid_local_attempt_rate": route_report(rows)["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": _rate(len(feasible), len(rows)),
        "regional_macro_non_noop_rate": _rate(len(regional_macro_non_noop), len(regional_macro)),
        "regional_macro_feasible_rate": _rate(len(regional_macro_feasible), len(regional_macro)),
        "no_op_count_by_strategy": counts["noop"],
        "infeasible_count_by_strategy": counts["infeasible"],
        "legalizer_limited_count": len(legalizer),
        "poor_targeting_count": len(poor),
        "runtime_proxy_per_case": dict(sorted(per_case_runtime.items())),
        "step7c_real_a_comparison": {
            "real_a_actual_safe_improvement_count": int(
                real_a_feasibility.get("actual_safe_improvement_count", 0)
            ),
            "real_a_official_like_cost_improving_count": 1,
            "real_a_official_like_hard_feasible_rate": float(
                real_a_feasibility.get("official_like_hard_feasible_rate", 0.0)
            ),
            "real_a_non_global_candidate_rate": float(
                real_a_feasibility.get("real_non_global_candidate_rate", 0.0)
            ),
        },
    }


def metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)
    return {
        "overall": _metric_distributions(rows),
        "by_strategy": {
            strategy: _metric_distributions(strategy_rows)
            for strategy, strategy_rows in sorted(by_strategy.items())
        },
        "per_candidate": [_metric_row(row) for row in rows],
    }


def pareto_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_id"])].append(row)
    per_case: dict[str, Any] = {}
    for case_id, case_rows in sorted(by_case.items()):
        front = _pareto_front(case_rows)
        per_case[str(case_id)] = {
            "front": [_pareto_row(row) for row in front],
            "front_size": len(front),
            "original_included": any(row["strategy"] == "original_layout" for row in case_rows),
        }
    return {
        "per_case": per_case,
        "original_inclusive_pareto_non_empty_count": sum(
            int(section["front_size"] > 0) for section in per_case.values()
        ),
        "original_inclusive_case_count": sum(
            int(section["original_included"]) for section in per_case.values()
        ),
    }


def failure_attribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": dict(Counter(str(row["failure_attribution"]) for row in rows)),
        "by_strategy": {
            strategy: dict(Counter(str(row["failure_attribution"]) for row in group))
            for strategy, group in _rows_by_strategy(rows).items()
        },
        "rows": [
            _compact_identity(row)
            | {
                "route_lane": row["route_lane"],
                "failure_attribution": row["failure_attribution"],
                "official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
                "official_like_cost_delta": row["official_like_cost_delta"],
                "hpwl_delta": row["hpwl_delta"],
                "bbox_area_delta": row["bbox_area_delta"],
                "no_op": row["no_op"],
            }
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def strategy_ablation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for strategy, group in _rows_by_strategy(rows).items():
        feasible = [row for row in group if row["after_route_official_like_hard_feasible"]]
        improving = [row for row in feasible if row["safe_improvement"]]
        cost_improving = [row for row in feasible if row["official_like_cost_improving"]]
        out[strategy] = {
            "candidate_count": len(group),
            "feasible_count": len(feasible),
            "improving_count": len(improving),
            "official_like_cost_improving_count": len(cost_improving),
            "improvement_density": _rate(len(improving), len(feasible)),
            "official_like_cost_improvement_density": _rate(len(cost_improving), len(feasible)),
            "no_op_count": sum(int(row["no_op"]) for row in group),
            "infeasible_count": len(group) - len(feasible),
            "mean_hpwl_delta": _mean(row["hpwl_delta"] for row in group),
            "mean_bbox_area_delta": _mean(row["bbox_area_delta"] for row in group),
            "mean_official_like_cost_delta": _mean(
                row["official_like_cost_delta"] for row in group
            ),
            "failure_attribution_counts": dict(
                Counter(str(row["failure_attribution"]) for row in group)
            ),
        }
    return out


def decision_for_step7c_real_b(
    feasibility: dict[str, Any],
    route: dict[str, Any],
    pareto: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    cost_improving = int(feasibility["official_like_cost_improving_count"])
    density = float(feasibility["official_like_cost_improvement_density"])
    if metrics["overall"]["official_like_cost_delta_distribution"]["count"] <= 0:
        return "inconclusive_due_to_metric_signal"
    if route["invalid_local_attempt_rate"] > 0.05:
        return "build_route_specific_legalizers"
    if float(feasibility["official_like_hard_feasible_rate"]) < 0.70:
        return "build_route_specific_legalizers"
    if pareto["original_inclusive_pareto_non_empty_count"] < feasibility["real_case_count"]:
        return "inconclusive_due_to_metric_signal"
    if int(feasibility["legalizer_limited_count"]) > int(feasibility["poor_targeting_count"]) * 1.5:
        return "build_route_specific_legalizers"
    if cost_improving >= 4 and density >= 0.10 and route["non_global_candidate_rate"] >= 0.70:
        return "promote_to_step7c_multi_iteration_sidecar"
    if cost_improving > 1:
        return "refine_metric_directed_generators"
    if feasibility["regional_macro_feasible_rate"] < 0.40:
        return "build_route_specific_legalizers"
    return "inconclusive_due_to_metric_signal"


def is_safe_improvement(row: dict[str, Any]) -> bool:
    if not bool(row["after_route_official_like_hard_feasible"]):
        return False
    if row["actual_locality_class"] == "global":
        return False
    return (
        _official_improving(row)
        or float(row["hpwl_delta"]) < -1e-9
        or float(row["bbox_area_delta"]) < -1e-9
    )


def failure_attribution(row: dict[str, Any]) -> str:
    if row["strategy"] == "original_layout":
        return "none"
    if row["descriptor_locality_class"] != "global" and row["actual_locality_class"] == "global":
        return "poor_targeting_collapsed_to_global"
    if row["report_only"]:
        return "global_report_only_not_repaired"
    if row["no_op"]:
        if str(row["construction_status"]).startswith("no_feasible"):
            return "legalizer_limited_no_feasible_real_edit"
        return "poor_targeting_no_effect"
    if not bool(row["after_route_official_like_hard_feasible"]):
        return "legalizer_limited_hard_infeasible"
    if row["official_like_cost_improving"]:
        return "none"
    if float(row["hpwl_delta"]) < -1e-9 or float(row["bbox_area_delta"]) < -1e-9:
        return "poor_targeting_proxy_only_improvement"
    return "poor_targeting_non_improving"


def _case_context(case: FloorSetCase, baseline: dict[int, Placement]) -> _CaseMetricContext:
    quality = _official_quality(case, baseline)
    return _CaseMetricContext(
        case=case,
        baseline=baseline,
        frame=frame_from_baseline(baseline),
        baseline_quality=quality,
        baseline_hpwl=hpwl_proxy(case, baseline),
        baseline_bbox_area=bbox_area(baseline),
        baseline_soft_total=soft_delta_components(case, baseline)["total"],
    )


def _build_strategy_candidate(ctx: _CaseMetricContext, strategy: str) -> _BuildResult:
    if strategy == "original_layout":
        return _BuildResult(
            dict(ctx.baseline), set(), set(), "original_baseline", ("original baseline",), 0, 1, 0.0
        )
    if strategy == "legacy_step7g_global_report":
        return _global_report_candidate(ctx)
    trials = _strategy_trials(ctx, strategy)
    feasible_trials = [trial for trial in trials if _hard_feasible(ctx, trial.edited)]
    selected = _select_metric_best(ctx, feasible_trials)
    if selected is None:
        return _BuildResult(
            dict(ctx.baseline),
            set(),
            _macro_closure_for_strategy(ctx.case, strategy),
            f"no_feasible_{strategy}",
            ("no official-like hard-feasible trial survived deterministic search",),
            len(trials),
            0,
            0.0,
        )
    status = f"constructed_{strategy}"
    notes = (
        selected.note,
        f"selected_official_like_cost_delta={selected.official_like_cost_delta:.6g}",
    )
    return _BuildResult(
        selected.edited,
        selected.changed,
        selected.macro_closure,
        status,
        notes,
        len(trials),
        len(feasible_trials),
        float(selected.official_like_cost_delta or 0.0),
    )


def _strategy_trials(ctx: _CaseMetricContext, strategy: str) -> list[_Trial]:
    movable = movable_blocks(ctx.case)
    if strategy == "hpwl_directed_local_nudge":
        groups = [[idx] for idx in _rank_blocks_by_metric_pressure(ctx, movable)[:14]]
        return _shift_trials(ctx, strategy, groups, vector_mode="hpwl", max_trials=90)
    if strategy == "bbox_shrink_nudge":
        groups = [[idx] for idx in _bbox_extreme_blocks(ctx, movable)[:16]]
        return _shift_trials(ctx, strategy, groups, vector_mode="bbox", max_trials=96)
    if strategy == "slack_fit_insertion":
        groups = [[idx] for idx in _rank_blocks_by_metric_pressure(ctx, movable)[:10]]
        return _shift_trials(ctx, strategy, groups, vector_mode="slack", max_trials=90)
    if strategy == "balanced_swap":
        return _balanced_swap_trials(ctx, movable, max_pairs=80)
    if strategy == "regional_metric_directed_shift":
        groups = _regional_metric_groups(ctx, movable)
        return _shift_trials(ctx, strategy, groups, vector_mode="regional", max_trials=100)
    if strategy == "macro_closure_metric_move":
        groups = _constraint_groups(ctx.case, movable)
        return _shift_trials(ctx, strategy, groups, vector_mode="macro", max_trials=90)
    raise ValueError(f"unknown metric-directed strategy: {strategy}")


def _select_metric_best(ctx: _CaseMetricContext, trials: list[_Trial]) -> _Trial | None:
    if not trials:
        return None
    shortlist = sorted(trials, key=lambda trial: trial.proxy_key)[:16]
    scored: list[_Trial] = []
    for trial in shortlist:
        delta = (
            _official_quality(ctx.case, trial.edited)["official_like_cost"]
            - ctx.baseline_quality["official_like_cost"]
        )
        scored.append(
            _Trial(
                trial.edited,
                trial.changed,
                trial.macro_closure,
                trial.note,
                trial.proxy_key,
                delta,
            )
        )
    return min(
        scored,
        key=lambda trial: (
            float(trial.official_like_cost_delta or 0.0),
            trial.proxy_key,
            sorted(trial.changed),
        ),
    )


def _shift_trials(
    ctx: _CaseMetricContext,
    strategy: str,
    groups: list[list[int]],
    *,
    vector_mode: str,
    max_trials: int,
) -> list[_Trial]:
    out: list[_Trial] = []
    for group in groups:
        group = [idx for idx in dict.fromkeys(group) if idx in ctx.baseline]
        if not group:
            continue
        for dx, dy in _candidate_vectors(ctx, group, vector_mode):
            edited = dict(ctx.baseline)
            for idx in group:
                x, y, w, h = edited[idx]
                edited[idx] = (x + dx, y + dy, w, h)
            changed = changed_blocks(ctx.baseline, edited)
            closure = set(group) if strategy != "macro_closure_metric_move" else set(group)
            out.append(
                _Trial(
                    edited,
                    changed,
                    closure,
                    f"{strategy}: shifted real block ids {group} by ({dx:.4g}, {dy:.4g})",
                    _proxy_key(ctx, edited, changed, prefer_bbox=(vector_mode == "bbox")),
                )
            )
            if len(out) >= max_trials:
                return out
    return out


def _candidate_vectors(
    ctx: _CaseMetricContext, group: list[int], mode: str
) -> list[tuple[float, float]]:
    if mode == "bbox":
        return _bbox_inward_vectors(ctx, group)
    cx, cy = _group_center(ctx.baseline, group)
    if mode in {"hpwl", "regional", "macro"}:
        tx, ty = _weighted_target_center(ctx.case, ctx.baseline, group)
    elif mode == "slack":
        tx, ty = _nearest_slack_target(ctx, group)
    else:
        tx, ty = ctx.frame.center
    dx = tx - cx
    dy = ty - cy
    norm = math.hypot(dx, dy)
    min_dim = min(min(ctx.baseline[idx][2], ctx.baseline[idx][3]) for idx in group)
    base = max(min(min_dim * 0.35, norm * 0.35 if norm > 0 else min_dim * 0.15), 0.05)
    primary = (0.0, 0.0) if norm <= 1e-9 else (dx / norm * base, dy / norm * base)
    vectors = [primary]
    for factor in (0.5, 1.0, 1.5):
        if abs(primary[0]) > 1e-12 or abs(primary[1]) > 1e-12:
            vectors.append((primary[0] * factor, primary[1] * factor))
    grid = max(base * 0.5, 0.05)
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        d = math.hypot(sx, sy)
        vectors.append((sx / d * grid, sy / d * grid))
    return _unique_vectors(vectors)


def _bbox_inward_vectors(ctx: _CaseMetricContext, group: list[int]) -> list[tuple[float, float]]:
    bbox = _bbox_from_placements(ctx.baseline.values())
    if bbox is None:
        return []
    bx0, by0, bx1, by1 = bbox
    gx, gy = _group_center(ctx.baseline, group)
    dx = -1.0 if abs(gx - bx1) < abs(gx - bx0) else 1.0
    dy = -1.0 if abs(gy - by1) < abs(gy - by0) else 1.0
    min_dim = min(min(ctx.baseline[idx][2], ctx.baseline[idx][3]) for idx in group)
    steps = [max(min_dim * factor, 0.05) for factor in (0.10, 0.20, 0.35, 0.50)]
    vectors: list[tuple[float, float]] = []
    for step in steps:
        vectors.extend([(dx * step, 0.0), (0.0, dy * step), (dx * step, dy * step)])
    return _unique_vectors(vectors)


def _balanced_swap_trials(
    ctx: _CaseMetricContext, movable: list[int], *, max_pairs: int
) -> list[_Trial]:
    ranked = _rank_blocks_by_metric_pressure(ctx, movable)[:18]
    pairs: list[tuple[float, int, int]] = []
    for left_idx, left in enumerate(ranked):
        for right in ranked[left_idx + 1 :]:
            if not _compatible_swap(ctx.baseline[left], ctx.baseline[right]):
                continue
            before = _block_metric_pressure(ctx, left) + _block_metric_pressure(ctx, right)
            after = _block_metric_pressure_if_swapped(ctx, left, right)
            pairs.append((after - before, left, right))
    out: list[_Trial] = []
    for _score, left, right in sorted(pairs)[:max_pairs]:
        edited = dict(ctx.baseline)
        lx, ly, lw, lh = edited[left]
        rx, ry, rw, rh = edited[right]
        edited[left] = (rx + (rw - lw) / 2.0, ry + (rh - lh) / 2.0, lw, lh)
        edited[right] = (lx + (lw - rw) / 2.0, ly + (lh - rh) / 2.0, rw, rh)
        changed = changed_blocks(ctx.baseline, edited)
        out.append(
            _Trial(
                edited,
                changed,
                set(changed),
                f"balanced_swap: swapped compatible real block ids {left} and {right}",
                _proxy_key(ctx, edited, changed),
            )
        )
    return out


def _global_report_candidate(ctx: _CaseMetricContext) -> _BuildResult:
    movable = movable_blocks(ctx.case)
    chosen = movable[: max(1, min(len(movable), round(ctx.case.block_count * 0.80)))]
    edited = dict(ctx.baseline)
    anchor_x = ctx.frame.xmin - max(ctx.frame.width * 0.05, 1.0)
    anchor_y = ctx.frame.ymin - max(ctx.frame.height * 0.05, 1.0)
    for offset, idx in enumerate(chosen):
        _x, _y, w, h = edited[idx]
        edited[idx] = (anchor_x + (offset % 2) * 0.1, anchor_y + (offset // 2 % 2) * 0.1, w, h)
    return _BuildResult(
        edited,
        set(chosen),
        set(chosen),
        "constructed_global_report_only",
        ("broad report-only baseline; never sent to bounded-local repair",),
        1,
        0,
        0.0,
    )


def _proxy_key(
    ctx: _CaseMetricContext,
    edited: dict[int, Placement],
    changed: set[int],
    *,
    prefer_bbox: bool = False,
) -> tuple[float, float, float, float]:
    hpwl_delta = hpwl_proxy(ctx.case, edited) - ctx.baseline_hpwl
    bbox_delta = bbox_area(edited) - ctx.baseline_bbox_area
    soft_delta = float(soft_delta_components(ctx.case, edited)["total"] - ctx.baseline_soft_total)
    disruption = sum(_distance(_center(ctx.baseline[idx]), _center(edited[idx])) for idx in changed)
    if prefer_bbox:
        return (bbox_delta, hpwl_delta, soft_delta, disruption)
    return (hpwl_delta, bbox_delta, soft_delta, disruption)


def _hard_feasible(ctx: _CaseMetricContext, edited: dict[int, Placement]) -> bool:
    return bool(
        official_like_hard_summary(ctx.case, edited, ctx.frame)["official_like_hard_feasible"]
    )


def _official_quality(case: FloorSetCase, placements: dict[int, Placement]) -> dict[str, float]:
    result = evaluate_positions(case, _positions_list(placements, case.block_count), runtime=1.0)
    quality = result.get("quality", {})
    return {
        "official_like_cost": float(quality.get("official_cost_raw", 0.0)),
        "hpwl_gap": float(quality.get("HPWLgap", 0.0)),
        "area_gap": float(quality.get("Areagap_bbox", 0.0)),
        "violations_relative": float(quality.get("Violationsrelative", 0.0)),
    }


def _official_improving(row: dict[str, Any]) -> bool:
    return (
        bool(row["after_route_official_like_hard_feasible"])
        and float(row["official_like_cost_delta"]) < -1e-9
    )


def _strategy_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = {
        key: Counter() for key in ("candidate", "feasible", "improving", "noop", "infeasible")
    }
    for row in rows:
        strategy = str(row["strategy"])
        buckets["candidate"][strategy] += 1
        if row["after_route_official_like_hard_feasible"]:
            buckets["feasible"][strategy] += 1
        else:
            buckets["infeasible"][strategy] += 1
        if row["safe_improvement"]:
            buckets["improving"][strategy] += 1
        if row["no_op"]:
            buckets["noop"][strategy] += 1
    return {key: dict(counter) for key, counter in buckets.items()}


def _metric_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hpwl_delta_distribution": _distribution(row["hpwl_delta"] for row in rows),
        "bbox_area_delta_distribution": _distribution(row["bbox_area_delta"] for row in rows),
        "soft_constraint_delta_distribution": _distribution(
            row["mib_group_boundary_soft_delta"]["total_delta"] for row in rows
        ),
        "official_like_cost_delta_distribution": _distribution(
            row["official_like_cost_delta"] for row in rows
        ),
        "official_like_hpwl_gap_delta_distribution": _distribution(
            row["official_like_hpwl_gap_delta"] for row in rows
        ),
        "official_like_area_gap_delta_distribution": _distribution(
            row["official_like_area_gap_delta"] for row in rows
        ),
        "official_like_violation_delta_distribution": _distribution(
            row["official_like_violation_delta"] for row in rows
        ),
    }


def _rank_blocks_by_metric_pressure(ctx: _CaseMetricContext, movable: list[int]) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for src, dst, weight in ctx.case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i in ctx.baseline and j in ctx.baseline:
            dist = _distance(_center(ctx.baseline[i]), _center(ctx.baseline[j])) * abs(
                float(weight)
            )
            scores[i] += dist
            scores[j] += dist
    for pin_idx, block_idx, weight in ctx.case.p2b_edges.tolist():
        p, b = int(pin_idx), int(block_idx)
        if b in ctx.baseline and 0 <= p < len(ctx.case.pins_pos):
            px, py = [float(v) for v in ctx.case.pins_pos[p].tolist()]
            scores[b] += _distance(_center(ctx.baseline[b]), (px, py)) * abs(float(weight))
    return sorted(movable, key=lambda idx: (-scores[idx], idx))


def _bbox_extreme_blocks(ctx: _CaseMetricContext, movable: list[int]) -> list[int]:
    bbox = _bbox_from_placements(ctx.baseline.values())
    if bbox is None:
        return movable
    x0, y0, x1, y1 = bbox
    rows = []
    for idx in movable:
        x, y, w, h = ctx.baseline[idx]
        pressure = max(
            max(0.0, 1.0 - abs(x - x0)),
            max(0.0, 1.0 - abs(y - y0)),
            max(0.0, 1.0 - abs((x + w) - x1)),
            max(0.0, 1.0 - abs((y + h) - y1)),
        )
        if pressure > 0.0:
            rows.append((idx, pressure, _block_metric_pressure(ctx, idx)))
    return [
        idx for idx, _pressure, _metric in sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))
    ]


def _regional_metric_groups(ctx: _CaseMetricContext, movable: list[int]) -> list[list[int]]:
    ranked = _rank_blocks_by_metric_pressure(ctx, movable)
    seeds = ranked[:8]
    groups = []
    for seed in seeds:
        sx, sy = _center(ctx.baseline[seed])
        nearest = sorted(
            movable,
            key=lambda idx: (_distance((sx, sy), _center(ctx.baseline[idx])), idx),
        )[: max(3, min(20, round(ctx.case.block_count * 0.16)))]
        groups.append(nearest)
    return groups


def _constraint_groups(case: FloorSetCase, movable: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for col in (ConstraintColumns.MIB, ConstraintColumns.CLUSTER):
        by_id: dict[int, list[int]] = defaultdict(list)
        for idx in movable:
            value = int(float(case.constraints[idx, col].item()))
            if value > 0:
                by_id[value].append(idx)
        groups.extend(sorted(members) for members in by_id.values() if members)
    groups.sort(key=lambda group: (-len(group), group[0]))
    return groups or ([movable[: min(4, len(movable))]] if movable else [])


def _macro_closure_for_strategy(case: FloorSetCase, strategy: str) -> set[int]:
    if strategy != "macro_closure_metric_move":
        return set()
    movable = movable_blocks(case)
    groups = _constraint_groups(case, movable)
    return set(groups[0]) if groups else set()


def _predicted_macro_size(strategy: str, block_count: int, result: _BuildResult) -> int:
    if strategy == "legacy_step7g_global_report":
        return max(1, round(block_count * 0.80))
    if strategy == "macro_closure_metric_move":
        return max(len(result.macro_closure), max(3, round(block_count * 0.30)))
    return len(result.macro_closure)


def _weighted_target_center(
    case: FloorSetCase, baseline: dict[int, Placement], group: list[int]
) -> tuple[float, float]:
    xs: list[tuple[float, float]] = []
    ys: list[tuple[float, float]] = []
    group_set = set(group)
    for src, dst, weight in case.b2b_edges.tolist():
        i = int(src)
        j = int(dst)
        w = abs(float(weight))
        if i in group_set and j in baseline and j not in group_set:
            x, y = _center(baseline[j])
            xs.append((x, w))
            ys.append((y, w))
        elif j in group_set and i in baseline and i not in group_set:
            x, y = _center(baseline[i])
            xs.append((x, w))
            ys.append((y, w))
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        p, b = int(pin_idx), int(block_idx)
        if b in group_set and 0 <= p < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[p].tolist()]
            w = abs(float(weight))
            xs.append((px, w))
            ys.append((py, w))
    if xs and ys:
        return _weighted_mean(xs), _weighted_mean(ys)
    return _group_center(baseline, group)


def _nearest_slack_target(ctx: _CaseMetricContext, group: list[int]) -> tuple[float, float]:
    gx, gy = _group_center(ctx.baseline, group)
    best: tuple[float, float, float] | None = None
    sample = min(min(ctx.baseline[idx][2], ctx.baseline[idx][3]) for idx in group) * 0.35
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            if sx == 0.0 and sy == 0.0:
                continue
            candidate = (gx + sx * sample, gy + sy * sample)
            score = _distance((gx, gy), candidate)
            if best is None or score < best[0]:
                best = (score, candidate[0], candidate[1])
    return (best[1], best[2]) if best is not None else (gx, gy)


def _compatible_swap(left: Placement, right: Placement) -> bool:
    la = left[2] * left[3]
    ra = right[2] * right[3]
    if abs(la - ra) / max(la, ra, 1e-9) > 0.25:
        return False
    return (
        abs(left[2] - right[2]) / max(left[2], right[2], 1e-9) <= 0.35
        and abs(left[3] - right[3]) / max(left[3], right[3], 1e-9) <= 0.35
    )


def _block_metric_pressure(ctx: _CaseMetricContext, idx: int) -> float:
    total = 0.0
    for src, dst, weight in ctx.case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if idx == i and j in ctx.baseline:
            total += _distance(_center(ctx.baseline[i]), _center(ctx.baseline[j])) * abs(
                float(weight)
            )
        elif idx == j and i in ctx.baseline:
            total += _distance(_center(ctx.baseline[i]), _center(ctx.baseline[j])) * abs(
                float(weight)
            )
    for pin_idx, block_idx, weight in ctx.case.p2b_edges.tolist():
        p, b = int(pin_idx), int(block_idx)
        if idx == b and 0 <= p < len(ctx.case.pins_pos):
            px, py = [float(v) for v in ctx.case.pins_pos[p].tolist()]
            total += _distance(_center(ctx.baseline[idx]), (px, py)) * abs(float(weight))
    return total


def _block_metric_pressure_if_swapped(ctx: _CaseMetricContext, left: int, right: int) -> float:
    edited = dict(ctx.baseline)
    lx, ly, lw, lh = edited[left]
    rx, ry, rw, rh = edited[right]
    edited[left] = (rx + (rw - lw) / 2.0, ry + (rh - lh) / 2.0, lw, lh)
    edited[right] = (lx + (lw - rw) / 2.0, ly + (lh - rh) / 2.0, rw, rh)
    temp = _CaseMetricContext(
        ctx.case,
        edited,
        ctx.frame,
        ctx.baseline_quality,
        ctx.baseline_hpwl,
        ctx.baseline_bbox_area,
        ctx.baseline_soft_total,
    )
    return _block_metric_pressure(temp, left) + _block_metric_pressure(temp, right)


def _route_for_expected(locality_class: str) -> str:
    if locality_class == "local":
        return "bounded_repair_pareto"
    if locality_class == "regional":
        return "region_repair_or_planner"
    if locality_class == "macro":
        return "macro_legalizer"
    return "global_route_not_local_selector"


def _rows_by_strategy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["strategy"])].append(row)
    return dict(sorted(out.items()))


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in rows if bool(row["after_route_official_like_hard_feasible"])]
    front = []
    for row in feasible:
        if not any(_dominates(other, row) for other in feasible if other is not row):
            front.append(row)
    return sorted(front, key=lambda row: (_objectives(row), row["candidate_id"]))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_obj = _objectives(left)
    right_obj = _objectives(right)
    return all(
        left_value <= right_value
        for left_value, right_value in zip(left_obj, right_obj, strict=False)
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(left_obj, right_obj, strict=False)
    )


def _objectives(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["official_like_cost_delta"]),
        float(row["hpwl_delta"]),
        float(row["bbox_area_delta"]),
        float(row["changed_block_fraction"]),
    )


def _pareto_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "candidate_id",
        "strategy",
        "descriptor_locality_class",
        "actual_locality_class",
        "actual_repair_mode",
        "route_lane",
        "after_route_official_like_hard_feasible",
        "hpwl_delta",
        "bbox_area_delta",
        "official_like_cost_delta",
        "mib_group_boundary_soft_delta",
        "changed_block_fraction",
    ]
    return {key: row[key] for key in keys}


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "candidate_id",
        "strategy",
        "actual_locality_class",
        "after_route_official_like_hard_feasible",
        "safe_improvement",
        "official_like_cost_improving",
        "hpwl_delta",
        "bbox_area_delta",
        "official_like_cost_delta",
        "mib_group_boundary_soft_delta",
    ]
    return {key: row[key] for key in keys}


def _compact_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "strategy": row["strategy"],
        "descriptor_locality_class": row["descriptor_locality_class"],
        "actual_locality_class": row["actual_locality_class"],
    }


def _positions_list(placements: dict[int, Placement], block_count: int) -> list[Placement]:
    return [placements[idx] for idx in range(block_count)]


def _group_center(placements: dict[int, Placement], group: list[int]) -> tuple[float, float]:
    centers = [_center(placements[idx]) for idx in group]
    return (
        sum(x for x, _y in centers) / max(len(centers), 1),
        sum(y for _x, y in centers) / max(len(centers), 1),
    )


def _center(box: Placement) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_w = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / max(total_w, 1e-9)


def _unique_vectors(vectors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for dx, dy in vectors:
        key = (round(dx, 6), round(dy, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((dx, dy))
    return out


def _distribution(values: Any) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "negative_count": 0,
            "positive_count": 0,
            "zero_count": 0,
        }
    return {
        "count": len(vals),
        "min": min(vals),
        "median": median(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "negative_count": sum(int(v < -1e-9) for v in vals),
        "positive_count": sum(int(v > 1e-9) for v in vals),
        "zero_count": sum(int(abs(v) <= 1e-9) for v in vals),
    }


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / max(len(vals), 1)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
