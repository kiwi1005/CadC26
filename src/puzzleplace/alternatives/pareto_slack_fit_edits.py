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
from puzzleplace.data import FloorSetCase
from puzzleplace.research.move_library import hpwl_proxy
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, _bbox_from_placements

STRATEGIES = (
    "original_layout",
    "hpwl_directed_local_nudge_expanded",
    "slack_fit_insertion_expanded",
    "nearby_slack_slot_insertion",
    "legacy_step7g_global_report",
)

BASELINE_STEP7C_REAL_B = {
    "official_like_cost_improving_count": 3,
    "official_like_cost_improvement_density": 0.05357142857142857,
    "hpwl_gain_to_official_like_conversion_rate": 0.25,
}

EPSILON_SENSITIVITY = (0.0, 1e-12, 1e-9, 1e-7)
DEFAULT_EPSILON = 1e-9


@dataclass(frozen=True)
class ParetoSlackFitCandidate:
    case_id: int
    candidate_id: str
    strategy: str
    variant_index: int
    descriptor_locality_class: str
    descriptor_repair_mode: str
    case: FloorSetCase
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    changed_blocks: tuple[int, ...]
    construction_status: str
    construction_notes: tuple[str, ...]
    construction_ms: float
    proxy_hpwl_delta: float
    proxy_bbox_delta: float
    proxy_soft_delta: float
    displacement_magnitude: float


@dataclass(frozen=True)
class _CaseContext:
    case: FloorSetCase
    baseline: dict[int, Placement]
    frame: PuzzleFrame
    baseline_hpwl: float
    baseline_bbox_area: float
    baseline_soft_total: int


@dataclass(frozen=True)
class _Probe:
    strategy: str
    edited: dict[int, Placement]
    changed: set[int]
    note: str
    proxy_objective: tuple[float, float, float, float]


def build_pareto_slack_fit_candidates(
    case_ids: list[int], cases_by_id: dict[int, FloorSetCase]
) -> list[ParetoSlackFitCandidate]:
    candidates: list[ParetoSlackFitCandidate] = []
    for case_id in case_ids:
        case = cases_by_id[case_id]
        baseline = placements_from_case(case)
        ctx = _case_context(case, baseline)
        case_candidates: list[ParetoSlackFitCandidate] = []
        case_candidates.append(_original_candidate(case_id, ctx))
        probes = _expanded_local_probes(ctx)
        for variant_index, probe in enumerate(probes):
            t0 = time.perf_counter()
            changed = tuple(sorted(changed_blocks(ctx.baseline, probe.edited)))
            proxy_hpwl = hpwl_proxy(case, probe.edited) - ctx.baseline_hpwl
            proxy_bbox = bbox_area(probe.edited) - ctx.baseline_bbox_area
            proxy_soft = float(
                soft_delta_components(case, probe.edited)["total"] - ctx.baseline_soft_total
            )
            case_candidates.append(
                ParetoSlackFitCandidate(
                    case_id=case_id,
                    candidate_id=f"case{case_id:03d}:{probe.strategy}:{variant_index:03d}",
                    strategy=probe.strategy,
                    variant_index=variant_index,
                    descriptor_locality_class="local",
                    descriptor_repair_mode="bounded_repair_pareto",
                    case=case,
                    baseline=baseline,
                    edited=probe.edited,
                    frame=ctx.frame,
                    changed_blocks=changed,
                    construction_status="constructed_local_slack_fit_probe",
                    construction_notes=(probe.note,),
                    construction_ms=(time.perf_counter() - t0) * 1000.0,
                    proxy_hpwl_delta=proxy_hpwl,
                    proxy_bbox_delta=proxy_bbox,
                    proxy_soft_delta=proxy_soft,
                    displacement_magnitude=_displacement(ctx.baseline, probe.edited, set(changed)),
                )
            )
        case_candidates.append(_global_report_candidate(case_id, ctx))
        candidates.extend(case_candidates)
    return candidates


def evaluate_pareto_slack_fit_candidate(candidate: ParetoSlackFitCandidate) -> dict[str, Any]:
    hard_before = official_like_hard_summary(candidate.case, candidate.edited, candidate.frame)
    changed = set(candidate.changed_blocks)
    regions = touched_region_count(candidate.edited, changed, candidate.frame)
    raw_fit = free_space_fit_ratio(candidate.edited, changed, candidate.frame)
    fit_ratio = min(raw_fit, 0.95) if hard_before["official_like_hard_feasible"] else raw_fit
    macro_size = (
        max(1, round(candidate.case.block_count * 0.80))
        if candidate.strategy == "legacy_step7g_global_report"
        else 0
    )
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.case.block_count,
        changed_block_count=len(changed),
        touched_region_count=regions,
        macro_closure_size=macro_size,
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
        "variant_index": candidate.variant_index,
        "descriptor_locality_class": candidate.descriptor_locality_class,
        "descriptor_repair_mode": candidate.descriptor_repair_mode,
        "actual_locality_class": actual_class,
        "actual_repair_mode": prediction["predicted_repair_mode"],
        "route_lane": repair["route_lane"],
        "report_only": repair["report_only"],
        "construction_status": candidate.construction_status,
        "construction_notes": list(candidate.construction_notes),
        "real_block_ids_changed": list(candidate.changed_blocks),
        "changed_block_count": len(changed),
        "changed_block_fraction": len(changed) / max(candidate.case.block_count, 1),
        "affected_region_count": regions,
        "macro_closure_size": macro_size,
        "free_space_fit_ratio": fit_ratio,
        "proxy_hpwl_delta": candidate.proxy_hpwl_delta,
        "proxy_bbox_delta": candidate.proxy_bbox_delta,
        "proxy_soft_delta": candidate.proxy_soft_delta,
        "displacement_magnitude": candidate.displacement_magnitude,
        **{f"before_repair_{key}": value for key, value in hard_before.items()},
        **{f"after_route_{key}": value for key, value in repair["after_hard"].items()},
        **metrics,
    }
    row["runtime_proxy_ms"] = candidate.construction_ms + float(metrics["metric_eval_ms"])
    row["no_op"] = len(changed) == 0 and candidate.strategy != "original_layout"
    row["official_like_cost_improving"] = _official_improving(row)
    row["hpwl_improving"] = float(row["hpwl_delta"]) < -DEFAULT_EPSILON
    row["safe_improvement"] = is_safe_improvement(row)
    row["objective_vector"] = objective_vector(row)
    row["failure_attribution"] = failure_attribution(row)
    return row


def evaluate_pareto_slack_fit_edits(
    candidates: list[ParetoSlackFitCandidate],
) -> list[dict[str, Any]]:
    return [evaluate_pareto_slack_fit_candidate(candidate) for candidate in candidates]


def objective_vector(row: dict[str, Any]) -> dict[str, Any]:
    soft = row.get("mib_group_boundary_soft_delta", {})
    return {
        "feasibility_rank": 0 if row["after_route_official_like_hard_feasible"] else 1,
        "route_rank": 1 if row["actual_locality_class"] == "global" else 0,
        "official_like_cost_delta": float(row["official_like_cost_delta"]),
        "hpwl_delta": float(row["hpwl_delta"]),
        "bbox_area_delta": float(row["bbox_area_delta"]),
        "soft_constraint_delta": float(soft.get("total_delta", 0.0)),
        "changed_block_fraction": float(row["changed_block_fraction"]),
    }


def objective_vector_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "candidate_id": row["candidate_id"],
            "strategy": row["strategy"],
            "actual_locality_class": row["actual_locality_class"],
            "objective_vector": row["objective_vector"],
            "official_like_cost_improving": row["official_like_cost_improving"],
            "hpwl_improving": row["hpwl_improving"],
        }
        for row in rows
    ]


def constrained_pareto_front(
    rows: list[dict[str, Any]], *, epsilon: float = DEFAULT_EPSILON
) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for row in rows:
        if not any(_dominates(other, row, epsilon=epsilon) for other in rows if other is not row):
            front.append(row)
    return sorted(front, key=lambda row: (_objective_tuple(row), row["candidate_id"]))


def pareto_front_report(
    rows: list[dict[str, Any]], *, epsilon: float = DEFAULT_EPSILON
) -> dict[str, Any]:
    by_case = _rows_by_case(rows)
    per_case: dict[str, Any] = {}
    for case_id, case_rows in by_case.items():
        front = constrained_pareto_front(case_rows, epsilon=epsilon)
        per_case[str(case_id)] = {
            "front_size": len(front),
            "front": [_pareto_row(row) for row in front],
            "original_included": any(row["strategy"] == "original_layout" for row in case_rows),
            "official_like_improving_in_front": sum(
                int(row["official_like_cost_improving"]) for row in front
            ),
        }
    return {
        "epsilon": epsilon,
        "per_case": per_case,
        "original_inclusive_pareto_non_empty_count": sum(
            int(section["front_size"] > 0) for section in per_case.values()
        ),
        "original_inclusive_case_count": sum(
            int(section["original_included"]) for section in per_case.values()
        ),
        "pareto_front_size_by_case": {
            case_id: section["front_size"] for case_id, section in per_case.items()
        },
    }


def dominance_report(
    rows: list[dict[str, Any]], *, epsilon: float = DEFAULT_EPSILON
) -> dict[str, Any]:
    by_case = _rows_by_case(rows)
    dominated_by_original: list[dict[str, Any]] = []
    hpwl_only_rejected: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    bbox_regression_hpwl = 0
    soft_regression_hpwl = 0
    for _case_id, case_rows in by_case.items():
        front_ids = {
            row["candidate_id"] for row in constrained_pareto_front(case_rows, epsilon=epsilon)
        }
        original = next((row for row in case_rows if row["strategy"] == "original_layout"), None)
        for row in case_rows:
            is_front = row["candidate_id"] in front_ids
            compact = _compact(row) | {"is_pareto_front": is_front}
            if is_front:
                retained.append(compact | {"reason": "non_dominated_constrained_pareto"})
            else:
                rejected.append(compact | {"reason": _rejection_reason(row, case_rows, epsilon)})
            if (
                original is not None
                and row is not original
                and _dominates(original, row, epsilon=epsilon)
            ):
                dominated_by_original.append(compact)
            if row["hpwl_improving"]:
                if float(row["bbox_area_delta"]) > epsilon:
                    bbox_regression_hpwl += 1
                if float(row["mib_group_boundary_soft_delta"]["total_delta"]) > epsilon:
                    soft_regression_hpwl += 1
                if not is_front:
                    hpwl_only_rejected.append(
                        compact | {"reason": _rejection_reason(row, case_rows, epsilon)}
                    )
    return {
        "epsilon": epsilon,
        "candidates_dominated_by_original_count": len(dominated_by_original),
        "candidates_dominated_by_original": dominated_by_original,
        "hpwl_only_winner_rejected_by_pareto_count": len(hpwl_only_rejected),
        "hpwl_only_winner_rejected_by_pareto": hpwl_only_rejected,
        "bbox_regression_among_hpwl_winners": bbox_regression_hpwl,
        "soft_regression_among_hpwl_winners": soft_regression_hpwl,
        "retained_candidates": retained,
        "rejected_candidates": rejected,
    }


def route_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row["actual_locality_class"]) for row in rows)
    local_rows = [row for row in rows if row["actual_repair_mode"] == "bounded_repair_pareto"]
    invalid_local = [
        row for row in local_rows if not row["before_repair_official_like_hard_feasible"]
    ]
    return {
        "route_count_by_class": dict(class_counts),
        "non_global_candidate_rate": _rate(len(rows) - class_counts["global"], len(rows)),
        "invalid_local_attempt_rate": _rate(len(invalid_local), len(local_rows)),
        "global_report_only_count": sum(int(row["report_only"]) for row in rows),
        "route_confusion": {
            key: dict(value)
            for key, value in _confusion(
                rows, "descriptor_locality_class", "actual_locality_class"
            ).items()
        },
    }


def metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metric_distributions(rows),
        "by_strategy": {
            strategy: _metric_distributions(group)
            for strategy, group in _rows_by_strategy(rows).items()
        },
        "per_candidate": [_metric_row(row) for row in rows],
    }


def failure_attribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": dict(Counter(str(row["failure_attribution"]) for row in rows)),
        "by_strategy": {
            strategy: dict(Counter(str(row["failure_attribution"]) for row in group))
            for strategy, group in _rows_by_strategy(rows).items()
        },
        "rows": [
            _compact(row)
            | {
                "failure_attribution": row["failure_attribution"],
                "route_lane": row["route_lane"],
            }
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def feasibility_and_improvement_report(
    rows: list[dict[str, Any]],
    pareto: dict[str, Any],
    dominance: dict[str, Any],
) -> dict[str, Any]:
    feasible = [row for row in rows if row["after_route_official_like_hard_feasible"]]
    official = [row for row in feasible if row["official_like_cost_improving"]]
    hpwl = [row for row in rows if row["hpwl_improving"]]
    per_case_runtime: dict[str, float] = defaultdict(float)
    for row in rows:
        per_case_runtime[str(row["case_id"])] += float(row["runtime_proxy_ms"])
    return {
        "real_case_count": len(_rows_by_case(rows)),
        "candidate_count_by_strategy": _count_by(rows, "strategy"),
        "feasible_count_by_strategy": _count_by(feasible, "strategy"),
        "route_count_by_class": dict(Counter(str(row["actual_locality_class"]) for row in rows)),
        "non_global_candidate_rate": _rate(
            sum(int(row["actual_locality_class"] != "global") for row in rows), len(rows)
        ),
        "invalid_local_attempt_rate": route_report(rows)["invalid_local_attempt_rate"],
        "official_like_hard_feasible_rate": _rate(len(feasible), len(rows)),
        "hpwl_improving_count": len(hpwl),
        "official_like_cost_improving_count": len(official),
        "hpwl_gain_to_official_like_conversion_rate": _rate(
            sum(int(row["official_like_cost_improving"]) for row in hpwl), len(hpwl)
        ),
        "improvement_density": _rate(
            sum(int(row["safe_improvement"]) for row in feasible), len(feasible)
        ),
        "official_like_cost_improvement_density": _rate(len(official), len(feasible)),
        "original_inclusive_pareto_non_empty_count": pareto[
            "original_inclusive_pareto_non_empty_count"
        ],
        "pareto_front_size_by_case": pareto["pareto_front_size_by_case"],
        "candidates_dominated_by_original_count": dominance[
            "candidates_dominated_by_original_count"
        ],
        "hpwl_only_winner_rejected_by_pareto_count": dominance[
            "hpwl_only_winner_rejected_by_pareto_count"
        ],
        "bbox_regression_among_hpwl_winners": dominance["bbox_regression_among_hpwl_winners"],
        "soft_regression_among_hpwl_winners": dominance["soft_regression_among_hpwl_winners"],
        "runtime_proxy_per_case": dict(sorted(per_case_runtime.items())),
    }


def sensitivity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eps_reports = []
    for eps in EPSILON_SENSITIVITY:
        pareto = pareto_front_report(rows, epsilon=eps)
        dominance = dominance_report(rows, epsilon=eps)
        front_ids = {
            row["candidate_id"]
            for section in pareto["per_case"].values()
            for row in section["front"]
        }
        eps_reports.append(
            {
                "epsilon": eps,
                "original_inclusive_pareto_non_empty_count": pareto[
                    "original_inclusive_pareto_non_empty_count"
                ],
                "total_front_size": sum(pareto["pareto_front_size_by_case"].values()),
                "official_like_improving_in_front": sum(
                    int(row["candidate_id"] in front_ids and row["official_like_cost_improving"])
                    for row in rows
                ),
                "dominated_by_original_count": dominance["candidates_dominated_by_original_count"],
                "hpwl_only_rejected_count": dominance["hpwl_only_winner_rejected_by_pareto_count"],
            }
        )
    return {
        "epsilon_source": (
            "numeric precision sensitivity around exact zero deltas; "
            "no hand-tuned bbox/soft threshold"
        ),
        "sensitivity_to_epsilon_or_tolerance": eps_reports,
    }


def decision_for_step7c_real_e(
    feasibility: dict[str, Any], route: dict[str, Any], pareto: dict[str, Any]
) -> str:
    official_count = int(feasibility["official_like_cost_improving_count"])
    density = float(feasibility["official_like_cost_improvement_density"])
    conversion = float(feasibility["hpwl_gain_to_official_like_conversion_rate"])
    hard_rate = float(feasibility["official_like_hard_feasible_rate"])
    if route["invalid_local_attempt_rate"] > 0.05:
        return "build_route_specific_legalizers"
    if hard_rate < 0.70:
        return "build_route_specific_legalizers"
    if pareto["original_inclusive_pareto_non_empty_count"] < feasibility["real_case_count"]:
        return "revisit_objective_vector_model"
    if official_count > BASELINE_STEP7C_REAL_B["official_like_cost_improving_count"]:
        if (
            density > BASELINE_STEP7C_REAL_B["official_like_cost_improvement_density"]
            and conversion > BASELINE_STEP7C_REAL_B["hpwl_gain_to_official_like_conversion_rate"]
            and route["non_global_candidate_rate"] >= 0.80
        ):
            return "promote_to_step7c_local_lane_iteration"
        return "refine_pareto_slack_fit_generator"
    if int(feasibility["hpwl_improving_count"]) == 0:
        return "pivot_to_coarse_region_planner"
    return "refine_pareto_slack_fit_generator"


def is_safe_improvement(row: dict[str, Any]) -> bool:
    if not bool(row["after_route_official_like_hard_feasible"]):
        return False
    if row["actual_locality_class"] == "global":
        return False
    return (
        _official_improving(row)
        or float(row["hpwl_delta"]) < -DEFAULT_EPSILON
        or float(row["bbox_area_delta"]) < -DEFAULT_EPSILON
    )


def failure_attribution(row: dict[str, Any]) -> str:
    if row["strategy"] == "original_layout":
        return "none"
    if row["report_only"]:
        return "global_report_only_not_repaired"
    if row["no_op"]:
        return "no_op_local_probe"
    if row["actual_locality_class"] == "global":
        return "route_collapse_globality"
    if not bool(row["after_route_official_like_hard_feasible"]):
        return "hard_infeasible_local_probe"
    if row["official_like_cost_improving"]:
        return "none"
    if row["hpwl_improving"]:
        if float(row["bbox_area_delta"]) > DEFAULT_EPSILON:
            return "hpwl_gain_cancelled_by_bbox"
        if float(row["mib_group_boundary_soft_delta"]["total_delta"]) > DEFAULT_EPSILON:
            return "hpwl_gain_cancelled_by_soft"
        return "hpwl_gain_not_official_like_improving"
    return "local_probe_non_improving"


def _expanded_local_probes(ctx: _CaseContext) -> list[_Probe]:
    movable = movable_blocks(ctx.case)
    ranked = _rank_blocks_by_metric_pressure(ctx, movable)[: min(20, len(movable))]
    all_by_strategy: dict[str, list[_Probe]] = defaultdict(list)
    for seed in ranked:
        all_by_strategy["hpwl_directed_local_nudge_expanded"].extend(
            _seed_direction_probes(ctx, seed, strategy="hpwl_directed_local_nudge_expanded")
        )
        all_by_strategy["slack_fit_insertion_expanded"].extend(
            _ring_slack_probes(ctx, seed, strategy="slack_fit_insertion_expanded", diagonal=False)
        )
        all_by_strategy["nearby_slack_slot_insertion"].extend(
            _ring_slack_probes(ctx, seed, strategy="nearby_slack_slot_insertion", diagonal=True)
        )
    selected: list[_Probe] = []
    per_strategy_limit = {
        "hpwl_directed_local_nudge_expanded": 24,
        "slack_fit_insertion_expanded": 28,
        "nearby_slack_slot_insertion": 20,
    }
    for strategy, probes in all_by_strategy.items():
        feasible_probes = [probe for probe in probes if _hard_feasible(ctx, probe.edited)]
        selected.extend(
            _select_probe_diverse_pareto(feasible_probes, limit=per_strategy_limit[strategy])
        )
    return selected


def _seed_direction_probes(ctx: _CaseContext, seed: int, *, strategy: str) -> list[_Probe]:
    target = _weighted_target_center(ctx.case, ctx.baseline, [seed])
    sx, sy = _center(ctx.baseline[seed])
    dx = target[0] - sx
    dy = target[1] - sy
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return []
    min_dim = min(ctx.baseline[seed][2], ctx.baseline[seed][3])
    vectors = [
        (dx / norm * min_dim * factor, dy / norm * min_dim * factor)
        for factor in (0.10, 0.18, 0.28, 0.40, 0.55, 0.75)
    ]
    return [_single_block_probe(ctx, seed, vx, vy, strategy) for vx, vy in vectors]


def _ring_slack_probes(
    ctx: _CaseContext, seed: int, *, strategy: str, diagonal: bool
) -> list[_Probe]:
    min_dim = min(ctx.baseline[seed][2], ctx.baseline[seed][3])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        directions += [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    probes: list[_Probe] = []
    for factor in (0.10, 0.18, 0.28, 0.40, 0.55):
        step = min_dim * factor
        for sx, sy in directions:
            d = math.hypot(sx, sy)
            probes.append(_single_block_probe(ctx, seed, sx / d * step, sy / d * step, strategy))
    return probes


def _single_block_probe(
    ctx: _CaseContext, seed: int, dx: float, dy: float, strategy: str
) -> _Probe:
    edited = dict(ctx.baseline)
    x, y, w, h = edited[seed]
    edited[seed] = (x + dx, y + dy, w, h)
    changed = changed_blocks(ctx.baseline, edited)
    proxy = _proxy_objective(ctx, edited, changed)
    return _Probe(
        strategy=strategy,
        edited=edited,
        changed=changed,
        note=f"{strategy}: moved real block {seed} by ({dx:.5g}, {dy:.5g})",
        proxy_objective=proxy,
    )


def _select_probe_diverse_pareto(probes: list[_Probe], *, limit: int) -> list[_Probe]:
    unique: dict[Any, _Probe] = {}
    for probe in probes:
        key = (
            len(probe.changed),
            tuple(
                sorted(
                    (
                        idx,
                        tuple(round(v, 8) for v in probe.edited[idx]),
                    )
                    for idx in probe.changed
                )
            ),
        )
        unique.setdefault(key, probe)
    candidates = list(unique.values())
    front = []
    for probe in candidates:
        if not any(_proxy_dominates(other, probe) for other in candidates if other is not probe):
            front.append(probe)
    ordered = sorted(front, key=lambda probe: (probe.proxy_objective, probe.note))
    if len(ordered) < limit:
        seen = {id(probe) for probe in ordered}
        ordered.extend(
            probe
            for probe in sorted(candidates, key=lambda probe: (probe.proxy_objective, probe.note))
            if id(probe) not in seen
        )
    return ordered[:limit]


def _proxy_dominates(left: _Probe, right: _Probe, *, eps: float = DEFAULT_EPSILON) -> bool:
    left_obj = left.proxy_objective
    right_obj = right.proxy_objective
    return all(a <= b + eps for a, b in zip(left_obj, right_obj, strict=False)) and any(
        a < b - eps for a, b in zip(left_obj, right_obj, strict=False)
    )


def _proxy_objective(
    ctx: _CaseContext, edited: dict[int, Placement], changed: set[int]
) -> tuple[float, float, float, float]:
    return (
        hpwl_proxy(ctx.case, edited) - ctx.baseline_hpwl,
        bbox_area(edited) - ctx.baseline_bbox_area,
        float(soft_delta_components(ctx.case, edited)["total"] - ctx.baseline_soft_total),
        _displacement(ctx.baseline, edited, changed),
    )


def _hard_feasible(ctx: _CaseContext, edited: dict[int, Placement]) -> bool:
    return bool(
        official_like_hard_summary(ctx.case, edited, ctx.frame)["official_like_hard_feasible"]
    )


def _case_context(case: FloorSetCase, baseline: dict[int, Placement]) -> _CaseContext:
    return _CaseContext(
        case=case,
        baseline=baseline,
        frame=frame_from_baseline(baseline),
        baseline_hpwl=hpwl_proxy(case, baseline),
        baseline_bbox_area=bbox_area(baseline),
        baseline_soft_total=soft_delta_components(case, baseline)["total"],
    )


def _original_candidate(case_id: int, ctx: _CaseContext) -> ParetoSlackFitCandidate:
    return ParetoSlackFitCandidate(
        case_id=case_id,
        candidate_id=f"case{case_id:03d}:original_layout",
        strategy="original_layout",
        variant_index=0,
        descriptor_locality_class="local",
        descriptor_repair_mode="bounded_repair_pareto",
        case=ctx.case,
        baseline=ctx.baseline,
        edited=dict(ctx.baseline),
        frame=ctx.frame,
        changed_blocks=(),
        construction_status="original_baseline",
        construction_notes=("original baseline",),
        construction_ms=0.0,
        proxy_hpwl_delta=0.0,
        proxy_bbox_delta=0.0,
        proxy_soft_delta=0.0,
        displacement_magnitude=0.0,
    )


def _global_report_candidate(case_id: int, ctx: _CaseContext) -> ParetoSlackFitCandidate:
    movable = movable_blocks(ctx.case)
    chosen = movable[: max(1, min(len(movable), round(ctx.case.block_count * 0.80)))]
    edited = dict(ctx.baseline)
    anchor_x = ctx.frame.xmin - max(ctx.frame.width * 0.05, 1.0)
    anchor_y = ctx.frame.ymin - max(ctx.frame.height * 0.05, 1.0)
    for offset, idx in enumerate(chosen):
        _x, _y, w, h = edited[idx]
        edited[idx] = (anchor_x + (offset % 3) * 0.1, anchor_y + (offset // 3 % 3) * 0.1, w, h)
    changed = changed_blocks(ctx.baseline, edited)
    return ParetoSlackFitCandidate(
        case_id=case_id,
        candidate_id=f"case{case_id:03d}:legacy_step7g_global_report",
        strategy="legacy_step7g_global_report",
        variant_index=0,
        descriptor_locality_class="global",
        descriptor_repair_mode="global_route_not_local_selector",
        case=ctx.case,
        baseline=ctx.baseline,
        edited=edited,
        frame=ctx.frame,
        changed_blocks=tuple(sorted(changed)),
        construction_status="constructed_global_report_only",
        construction_notes=("broad report-only baseline; never sent to bounded-local repair",),
        construction_ms=0.0,
        proxy_hpwl_delta=hpwl_proxy(ctx.case, edited) - ctx.baseline_hpwl,
        proxy_bbox_delta=bbox_area(edited) - ctx.baseline_bbox_area,
        proxy_soft_delta=float(
            soft_delta_components(ctx.case, edited)["total"] - ctx.baseline_soft_total
        ),
        displacement_magnitude=_displacement(ctx.baseline, edited, changed),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any], *, epsilon: float) -> bool:
    left_feasible = int(left["objective_vector"]["feasibility_rank"])
    right_feasible = int(right["objective_vector"]["feasibility_rank"])
    if left_feasible < right_feasible:
        return True
    if left_feasible > right_feasible:
        return False
    left_route = int(left["objective_vector"]["route_rank"])
    right_route = int(right["objective_vector"]["route_rank"])
    if left_route < right_route:
        return True
    if left_route > right_route:
        return False
    left_vec = _objective_tuple(left)[2:]
    right_vec = _objective_tuple(right)[2:]
    return all(a <= b + epsilon for a, b in zip(left_vec, right_vec, strict=False)) and any(
        a < b - epsilon for a, b in zip(left_vec, right_vec, strict=False)
    )


def _objective_tuple(row: dict[str, Any]) -> tuple[float, ...]:
    vector = row["objective_vector"]
    return (
        float(vector["feasibility_rank"]),
        float(vector["route_rank"]),
        float(vector["official_like_cost_delta"]),
        float(vector["hpwl_delta"]),
        float(vector["bbox_area_delta"]),
        float(vector["soft_constraint_delta"]),
        float(vector["changed_block_fraction"]),
    )


def _rejection_reason(row: dict[str, Any], case_rows: list[dict[str, Any]], epsilon: float) -> str:
    if any(
        other["strategy"] == "original_layout" and _dominates(other, row, epsilon=epsilon)
        for other in case_rows
    ):
        return "dominated_by_original"
    if row["hpwl_improving"] and (
        float(row["bbox_area_delta"]) > epsilon
        or float(row["mib_group_boundary_soft_delta"]["total_delta"]) > epsilon
    ):
        return "hpwl_gain_rejected_by_bbox_or_soft_tradeoff"
    if row["hpwl_improving"]:
        return "hpwl_gain_rejected_by_multi_objective_dominance"
    if not row["after_route_official_like_hard_feasible"]:
        return "infeasible_under_constrained_pareto"
    return "dominated_by_non_original_candidate"


def _rank_blocks_by_metric_pressure(ctx: _CaseContext, movable: list[int]) -> list[int]:
    return sorted(movable, key=lambda idx: (-_block_metric_pressure(ctx, idx), idx))


def _block_metric_pressure(ctx: _CaseContext, idx: int) -> float:
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


def _weighted_target_center(
    case: FloorSetCase, baseline: dict[int, Placement], group: list[int]
) -> tuple[float, float]:
    xs: list[tuple[float, float]] = []
    ys: list[tuple[float, float]] = []
    group_set = set(group)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
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
    bbox = _bbox_from_placements(baseline.values())
    if bbox is None:
        return _group_center(baseline, group)
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_w = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / max(total_w, 1e-9)


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


def _displacement(
    baseline: dict[int, Placement], edited: dict[int, Placement], changed: set[int]
) -> float:
    return sum(_distance(_center(baseline[idx]), _center(edited[idx])) for idx in changed)


def _official_improving(row: dict[str, Any]) -> bool:
    return (
        bool(row["after_route_official_like_hard_feasible"])
        and float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON
    )


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
    }


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
        "negative_count": sum(int(v < -DEFAULT_EPSILON) for v in vals),
        "positive_count": sum(int(v > DEFAULT_EPSILON) for v in vals),
        "zero_count": sum(int(abs(v) <= DEFAULT_EPSILON) for v in vals),
    }


def _rows_by_case(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(row["case_id"])].append(row)
    return dict(sorted(out.items()))


def _rows_by_strategy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["strategy"])].append(row)
    return dict(sorted(out.items()))


def _confusion(
    rows: list[dict[str, Any]], left_key: str, right_key: str
) -> dict[str, Counter[str]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[str(row[left_key])][str(row[right_key])] += 1
    return dict(sorted(matrix.items()))


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row[key]) for row in rows))


def _pareto_row(row: dict[str, Any]) -> dict[str, Any]:
    return _compact(row) | {
        "objective_vector": row["objective_vector"],
        "route_lane": row["route_lane"],
    }


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return _compact(row) | {
        "safe_improvement": row["safe_improvement"],
        "official_like_cost_improving": row["official_like_cost_improving"],
        "hpwl_improving": row["hpwl_improving"],
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "strategy": row["strategy"],
        "actual_locality_class": row["actual_locality_class"],
        "after_route_official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
        "official_like_cost_delta": row["official_like_cost_delta"],
        "hpwl_delta": row["hpwl_delta"],
        "bbox_area_delta": row["bbox_area_delta"],
        "soft_constraint_delta": row["mib_group_boundary_soft_delta"]["total_delta"],
        "changed_block_count": row["changed_block_count"],
    }


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0
