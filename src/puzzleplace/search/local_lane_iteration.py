from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import replace
from statistics import median
from typing import Any

from puzzleplace.alternatives.dominance_slack_preselection import (
    DEFAULT_RANK_BUDGET,
    preselect_dominance_slack_candidates,
)
from puzzleplace.alternatives.pareto_slack_fit_edits import (
    DEFAULT_EPSILON,
    ParetoSlackFitCandidate,
    _case_context,
    _expanded_local_probes,
    _global_report_candidate,
    _original_candidate,
    constrained_pareto_front,
    dominance_report,
    evaluate_pareto_slack_fit_edits,
    metric_report,
    objective_vector,
    pareto_front_report,
    route_report,
)
from puzzleplace.alternatives.real_placement_edits import (
    frame_from_baseline,
    metric_delta_report,
    official_like_hard_summary,
    placements_from_case,
    soft_delta_components,
)
from puzzleplace.data import FloorSetCase
from puzzleplace.research.move_library import hpwl_proxy
from puzzleplace.research.virtual_frame import Placement

DEFAULT_ITERATION_LIMIT = 3


def run_bounded_local_lane_iteration(
    case_ids: list[int],
    cases_by_id: dict[int, FloorSetCase],
    *,
    iteration_limit: int = DEFAULT_ITERATION_LIMIT,
    rank_budget: int = DEFAULT_RANK_BUDGET,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    filter_summaries: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    completed_iterations_by_case: dict[str, int] = {}
    runtime_proxy_per_case: dict[str, float] = defaultdict(float)

    for case_id in sorted(case_ids):
        case = cases_by_id[case_id]
        original = placements_from_case(case)
        current = dict(original)
        case_completed = 0
        stopped = False
        for iteration in range(iteration_limit):
            t0 = time.perf_counter()
            candidates = build_iteration_candidates(case_id, case, current, iteration=iteration)
            rows = evaluate_pareto_slack_fit_edits(candidates)
            rows = [_mark_iteration(row, iteration=iteration, phase="generated") for row in rows]
            candidate_rows.extend(rows)
            retained, filter_summary = preselect_dominance_slack_candidates(
                rows, rank_budget=rank_budget
            )
            retained = [
                _mark_iteration(row, iteration=iteration, phase="retained") for row in retained
            ]
            retained_rows.extend(retained)
            filter_summaries.append(
                {
                    "case_id": case_id,
                    "iteration": iteration,
                    **_compact_filter_summary(filter_summary),
                }
            )
            selected = select_current_pareto_local_edit(retained)
            cumulative_before = cumulative_metric_row(
                case_id, case, original, current, iteration=iteration, event="before_selection"
            )
            if selected is None:
                reason = skip_reason_for_iteration(rows, retained)
                cumulative_after = cumulative_before | {
                    "event": "stop_no_selected_edit",
                    "skip_reason": reason,
                }
                cumulative_rows.append(cumulative_after)
                traces.append(
                    _trace_row(
                        case_id=case_id,
                        iteration=iteration,
                        generated=rows,
                        retained=retained,
                        selected=None,
                        cumulative=cumulative_after,
                        skip_reason=reason,
                        runtime_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                )
                runtime_proxy_per_case[str(case_id)] += (time.perf_counter() - t0) * 1000.0
                stopped = True
                break
            selected_candidate = _candidate_by_id(candidates, str(selected["candidate_id"]))
            current = dict(selected_candidate.edited)
            selected = _mark_iteration(selected, iteration=iteration, phase="selected")
            selected_rows.append(selected)
            cumulative_after = cumulative_metric_row(
                case_id, case, original, current, iteration=iteration, event="after_selected_edit"
            )
            cumulative_after["selected_candidate_id"] = selected["candidate_id"]
            cumulative_after["selected_strategy"] = selected["strategy"]
            cumulative_after["per_iteration_hpwl_delta"] = selected["hpwl_delta"]
            cumulative_after["per_iteration_bbox_delta"] = selected["bbox_area_delta"]
            cumulative_after["per_iteration_soft_delta"] = selected[
                "mib_group_boundary_soft_delta"
            ]["total_delta"]
            cumulative_after["per_iteration_official_like_cost_delta"] = selected[
                "official_like_cost_delta"
            ]
            cumulative_rows.append(cumulative_after)
            traces.append(
                _trace_row(
                    case_id=case_id,
                    iteration=iteration,
                    generated=rows,
                    retained=retained,
                    selected=selected,
                    cumulative=cumulative_after,
                    skip_reason=None,
                    runtime_ms=(time.perf_counter() - t0) * 1000.0,
                )
            )
            case_completed += 1
            runtime_proxy_per_case[str(case_id)] += (time.perf_counter() - t0) * 1000.0
        if not stopped and iteration_limit == 0:
            cumulative_rows.append(
                cumulative_metric_row(
                    case_id, case, original, current, iteration=0, event="no_iterations"
                )
            )
        completed_iterations_by_case[str(case_id)] = case_completed

    reports = build_iteration_reports(
        traces=traces,
        candidate_rows=candidate_rows,
        retained_rows=retained_rows,
        selected_rows=selected_rows,
        filter_summaries=filter_summaries,
        cumulative_rows=cumulative_rows,
        real_case_count=len(case_ids),
        iteration_limit=iteration_limit,
        runtime_proxy_per_case=dict(sorted(runtime_proxy_per_case.items())),
        completed_iterations_by_case=completed_iterations_by_case,
    )
    reports["decision"] = decision_for_local_iter0(reports)
    return reports


def build_iteration_candidates(
    case_id: int, case: FloorSetCase, current: dict[int, Placement], *, iteration: int
) -> list[ParetoSlackFitCandidate]:
    ctx = _case_context(case, current)
    candidates: list[ParetoSlackFitCandidate] = []
    candidates.append(
        _retag_candidate(_original_candidate(case_id, ctx), iteration=iteration, variant_index=0)
    )
    for variant_index, probe in enumerate(_expanded_local_probes(ctx)):
        changed = tuple(sorted(_changed_blocks(ctx.baseline, probe.edited)))
        candidates.append(
            ParetoSlackFitCandidate(
                case_id=case_id,
                candidate_id=f"case{case_id:03d}:iter{iteration:02d}:{probe.strategy}:{variant_index:03d}",
                strategy=probe.strategy,
                variant_index=variant_index,
                descriptor_locality_class="local",
                descriptor_repair_mode="bounded_repair_pareto",
                case=case,
                baseline=current,
                edited=probe.edited,
                frame=ctx.frame,
                changed_blocks=changed,
                construction_status="constructed_sequential_local_slack_probe",
                construction_notes=(f"iter{iteration}: {probe.note}",),
                construction_ms=0.0,
                proxy_hpwl_delta=hpwl_proxy(case, probe.edited) - ctx.baseline_hpwl,
                proxy_bbox_delta=_bbox_proxy(case, ctx.baseline, probe.edited),
                proxy_soft_delta=float(
                    soft_delta_components(case, probe.edited)["total"] - ctx.baseline_soft_total
                ),
                displacement_magnitude=_displacement(ctx.baseline, probe.edited, set(changed)),
            )
        )
    candidates.append(
        _retag_candidate(
            _global_report_candidate(case_id, ctx),
            iteration=iteration,
            variant_index=len(candidates),
        )
    )
    return candidates


def select_current_pareto_local_edit(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    front_ids = {row["candidate_id"] for row in constrained_pareto_front(rows)}
    eligible = [
        row
        for row in rows
        if row["candidate_id"] in front_ids
        and row["strategy"] not in {"original_layout", "legacy_step7g_global_report"}
        and row["actual_locality_class"] == "local"
        and row["actual_repair_mode"] == "bounded_repair_pareto"
        and bool(row["after_route_official_like_hard_feasible"])
        and not bool(row["report_only"])
        and float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            float(row["official_like_cost_delta"]),
            float(row["hpwl_delta"]),
            float(row["bbox_area_delta"]),
            float(row["mib_group_boundary_soft_delta"]["total_delta"]),
            str(row["candidate_id"]),
        ),
    )


def skip_reason_for_iteration(
    generated_rows: list[dict[str, Any]], retained_rows: list[dict[str, Any]]
) -> str:
    local_generated = [
        row
        for row in generated_rows
        if row["strategy"] not in {"original_layout", "legacy_step7g_global_report"}
    ]
    if not local_generated:
        return "candidate_starvation_no_local_generated"
    local_retained = [
        row
        for row in retained_rows
        if row["strategy"] not in {"original_layout", "legacy_step7g_global_report"}
    ]
    if not local_retained:
        return "candidate_starvation_no_preselection_retained"
    feasible = [row for row in local_retained if row["after_route_official_like_hard_feasible"]]
    if not feasible:
        return "sequential_feasibility_collapse"
    if not any(float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON for row in feasible):
        return "no_official_like_improving_local_candidate"
    return "no_current_pareto_acceptable_local_edit"


def cumulative_metric_row(
    case_id: int,
    case: FloorSetCase,
    original: dict[int, Placement],
    current: dict[int, Placement],
    *,
    iteration: int,
    event: str,
) -> dict[str, Any]:
    frame = frame_from_baseline(original)
    hard = official_like_hard_summary(case, current, frame)
    metrics = metric_delta_report(case, original, current)
    row = {
        "case_id": case_id,
        "iteration": iteration,
        "event": event,
        "candidate_id": f"case{case_id:03d}:iter{iteration:02d}:current_state",
        "strategy": "current_sequential_state",
        "actual_locality_class": "local",
        "actual_repair_mode": "bounded_repair_pareto",
        "route_lane": "current_state",
        "report_only": False,
        "changed_block_count": len(_changed_blocks(original, current)),
        "changed_block_fraction": len(_changed_blocks(original, current))
        / max(case.block_count, 1),
        "after_route_official_like_hard_feasible": hard["official_like_hard_feasible"],
        "mib_group_boundary_soft_delta": metrics["mib_group_boundary_soft_delta"],
        **metrics,
    }
    row["official_like_cost_improving"] = (
        bool(row["after_route_official_like_hard_feasible"])
        and float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON
    )
    row["hpwl_improving"] = float(row["hpwl_delta"]) < -DEFAULT_EPSILON
    row["safe_improvement"] = row["official_like_cost_improving"] or row["hpwl_improving"]
    row["objective_vector"] = objective_vector(row)
    return row


def build_iteration_reports(
    *,
    traces: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    retained_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    filter_summaries: list[dict[str, Any]],
    cumulative_rows: list[dict[str, Any]],
    real_case_count: int,
    iteration_limit: int,
    runtime_proxy_per_case: dict[str, float],
    completed_iterations_by_case: dict[str, int],
) -> dict[str, Any]:
    current_pareto_by_iter = _pareto_non_empty_by_iteration(retained_rows)
    original_pareto_rows = cumulative_rows + _original_anchor_rows_for_iterations(cumulative_rows)
    original_pareto_by_iter = _pareto_non_empty_by_iteration(original_pareto_rows)
    route_by_iter = _route_count_by_iteration(retained_rows)
    filter_reason_counts = Counter(
        reason
        for summary in filter_summaries
        for reason, count in summary["filter_reason_counts"].items()
        for _ in range(int(count))
    )
    selected_by_iter = Counter(str(row["iteration"]) for row in selected_rows)
    skipped = Counter(row["skip_reason"] for row in traces if row.get("skip_reason"))
    retained_non_anchor = [
        row
        for row in retained_rows
        if row["strategy"] not in {"original_layout", "legacy_step7g_global_report"}
    ]
    retained_dominance = dominance_report(retained_rows) if retained_rows else _empty_dominance()
    original_pareto_rows = cumulative_rows + _original_anchor_rows_for_iterations(cumulative_rows)
    original_dominance = (
        dominance_report(original_pareto_rows) if original_pareto_rows else _empty_dominance()
    )
    generated_by_iter = _count_by_iteration(candidate_rows)
    retained_by_iter = _count_by_iteration(retained_rows)
    selected_metric = metric_report(selected_rows) if selected_rows else _empty_metric_report()
    cumulative_metric = (
        metric_report(cumulative_rows) if cumulative_rows else _empty_metric_report()
    )
    hard_by_iter = _hard_feasible_rate_by_iteration(retained_rows)
    invalid_local = (
        route_report(retained_rows)["invalid_local_attempt_rate"] if retained_rows else 0.0
    )
    sequential_feasibility_collapse = sum(
        int(row.get("skip_reason") == "sequential_feasibility_collapse") for row in traces
    )
    metric_regression_after_application = sum(
        int(
            row.get("event") == "after_selected_edit"
            and float(row["official_like_cost_delta"]) > DEFAULT_EPSILON
        )
        for row in cumulative_rows
    )
    reports = {
        "iteration_trace": {
            "real_case_count": real_case_count,
            "configured_iteration_limit": iteration_limit,
            "completed_iterations_by_case": completed_iterations_by_case,
            "rows": traces,
        },
        "candidate_report": {
            "real_case_count": real_case_count,
            "configured_iteration_limit": iteration_limit,
            "candidates_generated_by_iteration": generated_by_iter,
            "candidates_retained_by_preselection_by_iteration": retained_by_iter,
            "selected_edits_by_iteration": dict(sorted(selected_by_iter.items())),
            "skipped_cases_by_reason": dict(sorted(skipped.items())),
            "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
            "candidate_starvation_count": sum(
                count
                for reason, count in skipped.items()
                if str(reason).startswith("candidate_starvation")
            ),
            "generated_candidate_count": len(candidate_rows),
            "retained_candidate_count": len(retained_rows),
            "selected_edit_count": len(selected_rows),
            "retained_non_anchor_count": len(retained_non_anchor),
            "rows": [_candidate_summary(row) for row in retained_rows],
        },
        "route_report": {
            **route_report(retained_rows),
            "route_count_by_iteration": route_by_iter,
        },
        "feasibility_report": {
            "real_case_count": real_case_count,
            "configured_iteration_limit": iteration_limit,
            "official_like_hard_feasible_rate_by_iteration": hard_by_iter,
            "invalid_local_attempt_rate": invalid_local,
            "sequential_feasibility_collapse_count": sequential_feasibility_collapse,
            "current_state_hard_feasible_count": sum(
                int(bool(row["after_route_official_like_hard_feasible"]))
                for row in cumulative_rows
                if row.get("event") in {"after_selected_edit", "stop_no_selected_edit"}
            ),
        },
        "metric_report": {
            "selected_edit_metric_report": selected_metric,
            "cumulative_state_metric_report": cumulative_metric,
            "per_iteration_hpwl_delta": _distribution(
                row.get("per_iteration_hpwl_delta", 0.0)
                for row in cumulative_rows
                if row.get("event") == "after_selected_edit"
            ),
            "per_iteration_bbox_delta": _distribution(
                row.get("per_iteration_bbox_delta", 0.0)
                for row in cumulative_rows
                if row.get("event") == "after_selected_edit"
            ),
            "per_iteration_soft_delta": _distribution(
                row.get("per_iteration_soft_delta", 0.0)
                for row in cumulative_rows
                if row.get("event") == "after_selected_edit"
            ),
            "per_iteration_official_like_cost_delta": _distribution(
                row.get("per_iteration_official_like_cost_delta", 0.0)
                for row in cumulative_rows
                if row.get("event") == "after_selected_edit"
            ),
            "cumulative_hpwl_delta_by_case": _latest_cumulative_by_case(
                cumulative_rows, "hpwl_delta"
            ),
            "cumulative_bbox_delta_by_case": _latest_cumulative_by_case(
                cumulative_rows, "bbox_area_delta"
            ),
            "cumulative_soft_delta_by_case": _latest_cumulative_soft_by_case(cumulative_rows),
            "cumulative_official_like_cost_delta_by_case": _latest_cumulative_by_case(
                cumulative_rows, "official_like_cost_delta"
            ),
            "runtime_proxy_per_case": runtime_proxy_per_case,
        },
        "pareto_report": {
            "current_inclusive_pareto_non_empty_count_by_iteration": current_pareto_by_iter,
            "original_inclusive_pareto_non_empty_count_by_iteration": original_pareto_by_iter,
            "dominated_by_current_count": retained_dominance[
                "candidates_dominated_by_original_count"
            ],
            "dominated_by_original_count": original_dominance[
                "candidates_dominated_by_original_count"
            ],
            "current_pareto_front": pareto_front_report(retained_rows) if retained_rows else {},
            "original_pareto_front": (
                pareto_front_report(original_pareto_rows) if original_pareto_rows else {}
            ),
        },
        "failure_attribution": {
            "skipped_cases_by_reason": dict(sorted(skipped.items())),
            "candidate_starvation_count": sum(
                count
                for reason, count in skipped.items()
                if str(reason).startswith("candidate_starvation")
            ),
            "sequential_feasibility_collapse_count": sequential_feasibility_collapse,
            "metric_regression_after_application_count": metric_regression_after_application,
            "candidate_failure_reason_counts": dict(
                Counter(row.get("failure_attribution", "none") for row in retained_rows)
            ),
        },
    }
    return reports


def decision_for_local_iter0(reports: dict[str, Any]) -> str:
    candidate = reports["candidate_report"]
    feasible = reports["feasibility_report"]
    metric = reports["metric_report"]
    pareto = reports["pareto_report"]
    real_case_count = int(candidate["real_case_count"])
    selected_count = int(candidate["selected_edit_count"])
    improved_cases = sum(
        int(float(delta) < -DEFAULT_EPSILON)
        for delta in metric["cumulative_official_like_cost_delta_by_case"].values()
    )
    if float(feasible["invalid_local_attempt_rate"]) > 0.05:
        return "add_route_specific_local_legalizer"
    if int(feasible["sequential_feasibility_collapse_count"]) > 0:
        return "add_route_specific_local_legalizer"
    if int(candidate["candidate_starvation_count"]) >= max(1, real_case_count // 2):
        return "pivot_to_coarse_region_planner"
    if selected_count == 0:
        return "revisit_local_lane_iteration_assumptions"
    if int(pareto["original_inclusive_pareto_non_empty_count_by_iteration"].get("0", 0)) == 0:
        return "inconclusive_due_to_sequential_instability"
    promotion_floor = max(4, real_case_count // 2)
    if improved_cases >= promotion_floor and selected_count >= promotion_floor:
        return "promote_to_step7c_local_lane_search"
    if improved_cases > 0:
        return "refine_sequential_local_preselection"
    return "revisit_local_lane_iteration_assumptions"


def _retag_candidate(
    candidate: ParetoSlackFitCandidate, *, iteration: int, variant_index: int
) -> ParetoSlackFitCandidate:
    suffix = candidate.strategy
    return replace(
        candidate,
        candidate_id=f"case{candidate.case_id:03d}:iter{iteration:02d}:{suffix}",
        variant_index=variant_index,
        construction_notes=tuple(
            f"iter{iteration}: {note}" for note in candidate.construction_notes
        ),
    )


def _mark_iteration(row: dict[str, Any], *, iteration: int, phase: str) -> dict[str, Any]:
    copied = dict(row)
    copied["iteration"] = iteration
    copied["iteration_phase"] = phase
    return copied


def _candidate_by_id(
    candidates: list[ParetoSlackFitCandidate], candidate_id: str
) -> ParetoSlackFitCandidate:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _trace_row(
    *,
    case_id: int,
    iteration: int,
    generated: list[dict[str, Any]],
    retained: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    cumulative: dict[str, Any],
    skip_reason: str | None,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "iteration": iteration,
        "generated_candidate_count": len(generated),
        "retained_candidate_count": len(retained),
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_strategy": selected["strategy"] if selected else None,
        "skip_reason": skip_reason,
        "route_count": dict(Counter(str(row["actual_locality_class"]) for row in retained)),
        "current_official_like_cost_delta": cumulative["official_like_cost_delta"],
        "current_hpwl_delta": cumulative["hpwl_delta"],
        "current_bbox_delta": cumulative["bbox_area_delta"],
        "current_soft_delta": cumulative["mib_group_boundary_soft_delta"]["total_delta"],
        "current_hard_feasible": cumulative["after_route_official_like_hard_feasible"],
        "runtime_proxy_ms": runtime_ms,
    }


def _compact_filter_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_candidate_count": summary["source_candidate_count_from_E"],
        "retained_candidate_count": summary["retained_candidate_count"],
        "filtered_candidate_count": summary["filtered_candidate_count"],
        "retained_fraction": summary["retained_fraction"],
        "filter_reason_counts": summary["filter_reason_counts"],
    }


def _candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "iteration": row["iteration"],
        "candidate_id": row["candidate_id"],
        "strategy": row["strategy"],
        "actual_locality_class": row["actual_locality_class"],
        "route_lane": row["route_lane"],
        "filter_reason": row.get("filter_reason", "unlabeled"),
        "after_route_official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
        "official_like_cost_delta": row["official_like_cost_delta"],
        "hpwl_delta": row["hpwl_delta"],
        "bbox_area_delta": row["bbox_area_delta"],
        "soft_constraint_delta": row["mib_group_boundary_soft_delta"]["total_delta"],
        "official_like_cost_improving": row["official_like_cost_improving"],
    }


def _pareto_non_empty_by_iteration(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for iteration, group in _group_by_iteration(rows).items():
        out[str(iteration)] = pareto_front_report(group)[
            "original_inclusive_pareto_non_empty_count"
        ]
    return dict(sorted(out.items()))


def _route_count_by_iteration(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        str(iteration): dict(Counter(str(row["actual_locality_class"]) for row in group))
        for iteration, group in sorted(_group_by_iteration(rows).items())
    }


def _hard_feasible_rate_by_iteration(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for iteration, group in _group_by_iteration(rows).items():
        out[str(iteration)] = sum(
            int(bool(row["after_route_official_like_hard_feasible"])) for row in group
        ) / max(len(group), 1)
    return dict(sorted(out.items()))


def _count_by_iteration(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["iteration"]) for row in rows).items()))


def _group_by_iteration(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(row.get("iteration", 0))].append(row)
    return dict(sorted(out.items()))


def _original_anchor_rows_for_iterations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        case_id = int(row["case_id"])
        iteration = int(row.get("iteration", 0))
        key = (case_id, iteration)
        if key in seen:
            continue
        seen.add(key)
        anchor = dict(row)
        anchor.update(
            {
                "candidate_id": f"case{case_id:03d}:iter{iteration:02d}:original_anchor",
                "strategy": "original_layout",
                "event": "original_anchor",
                "changed_block_count": 0,
                "changed_block_fraction": 0.0,
                "hpwl_delta": 0.0,
                "bbox_area_delta": 0.0,
                "official_like_cost_delta": 0.0,
                "official_like_hpwl_gap_delta": 0.0,
                "official_like_area_gap_delta": 0.0,
                "official_like_violation_delta": 0.0,
                "mib_group_boundary_soft_delta": {
                    "mib_delta": 0,
                    "grouping_delta": 0,
                    "boundary_delta": 0,
                    "total_delta": 0,
                },
                "official_like_cost_improving": False,
                "hpwl_improving": False,
                "safe_improvement": False,
            }
        )
        anchor["objective_vector"] = objective_vector(anchor)
        anchors.append(anchor)
    return anchors


def _latest_cumulative_by_case(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        latest[int(row["case_id"])] = row
    return {str(case_id): float(row[key]) for case_id, row in sorted(latest.items())}


def _latest_cumulative_soft_by_case(rows: list[dict[str, Any]]) -> dict[str, float]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        latest[int(row["case_id"])] = row
    return {
        str(case_id): float(row["mib_group_boundary_soft_delta"]["total_delta"])
        for case_id, row in sorted(latest.items())
    }


def _empty_dominance() -> dict[str, Any]:
    return {"candidates_dominated_by_original_count": 0}


def _empty_metric_report() -> dict[str, Any]:
    return {"overall": {}, "by_strategy": {}, "per_candidate": []}


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


def _changed_blocks(
    baseline: dict[int, Placement], edited: dict[int, Placement], *, eps: float = 1e-9
) -> set[int]:
    return {
        idx
        for idx, before in baseline.items()
        if idx in edited
        and any(abs(a - b) > eps for a, b in zip(before, edited[idx], strict=False))
    }


def _bbox_proxy(
    case: FloorSetCase, baseline: dict[int, Placement], edited: dict[int, Placement]
) -> float:
    del case
    from puzzleplace.alternatives.real_placement_edits import bbox_area

    return bbox_area(edited) - bbox_area(baseline)


def _center(box: Placement) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _displacement(
    baseline: dict[int, Placement], edited: dict[int, Placement], changed: set[int]
) -> float:
    import math

    return sum(
        math.hypot(
            _center(baseline[idx])[0] - _center(edited[idx])[0],
            _center(baseline[idx])[1] - _center(edited[idx])[1],
        )
        for idx in changed
    )
