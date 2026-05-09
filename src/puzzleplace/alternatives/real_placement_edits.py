from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.research.move_library import (
    grouping_violation_count,
    hpwl_proxy,
    mib_violation_count,
    soft_boundary_violations,
)
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, _bbox_from_placements

SELECTED_FAMILIES = {
    "original_layout",
    "vacancy_aware_local_insertion",
    "adjacent_region_reassignment",
    "mib_group_closure_macro",
    "legacy_step7g_global_move",
}


@dataclass(frozen=True)
class RealPlacementEditCandidate:
    case_id: int
    candidate_id: str
    family: str
    descriptor_locality_class: str
    descriptor_repair_mode: str
    case: FloorSetCase
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    changed_blocks: tuple[int, ...]
    macro_closure_blocks: tuple[int, ...]
    predicted_macro_closure_size: int
    route_lane: str
    construction_status: str
    construction_notes: tuple[str, ...]
    construction_ms: float


def build_real_edit_candidates(
    step7h_predictions: list[dict[str, Any]],
    cases_by_id: dict[int, FloorSetCase],
) -> list[RealPlacementEditCandidate]:
    out: list[RealPlacementEditCandidate] = []
    for descriptor in step7h_predictions:
        family = str(descriptor["family"])
        if family not in SELECTED_FAMILIES:
            continue
        case_id = int(descriptor["case_id"])
        if case_id not in cases_by_id:
            continue
        case = cases_by_id[case_id]
        baseline = placements_from_case(case)
        frame = frame_from_baseline(baseline)
        t0 = time.perf_counter()
        edited, changed, macro_blocks, status, notes = apply_real_layout_edit(
            case,
            family,
            baseline,
            frame,
            descriptor=descriptor,
        )
        out.append(
            RealPlacementEditCandidate(
                case_id=case_id,
                candidate_id=str(descriptor["candidate_id"]),
                family=family,
                descriptor_locality_class=str(descriptor["predicted_locality_class"]),
                descriptor_repair_mode=str(descriptor["predicted_repair_mode"]),
                case=case,
                baseline=baseline,
                edited=edited,
                frame=frame,
                changed_blocks=tuple(sorted(changed_blocks(baseline, edited))),
                macro_closure_blocks=tuple(sorted(macro_blocks)),
                predicted_macro_closure_size=max(
                    int(descriptor.get("predicted_macro_closure_size", 0)),
                    len(macro_blocks),
                ),
                route_lane=_route_lane(family),
                construction_status=status,
                construction_notes=tuple(notes),
                construction_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )
    return out


def placements_from_case(case: FloorSetCase) -> dict[int, Placement]:
    placements: dict[int, Placement] = {}
    for idx, box in enumerate(positions_from_case_targets(case)):
        x, y, w, h = (float(v) for v in box)
        placements[idx] = (x, y, w, h)
    return placements


def frame_from_baseline(
    placements: dict[int, Placement], *, margin_ratio: float = 0.02
) -> PuzzleFrame:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return PuzzleFrame(0.0, 0.0, 1.0, 1.0, density=1.0, variant="baseline_bbox")
    xmin, ymin, xmax, ymax = bbox
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    mx = w * margin_ratio
    my = h * margin_ratio
    return PuzzleFrame(
        xmin - mx,
        ymin - my,
        xmax + mx,
        ymax + my,
        density=1.0,
        variant="baseline_bbox_margin",
    )


def apply_real_layout_edit(
    case: FloorSetCase,
    family: str,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    descriptor: dict[str, Any] | None = None,
) -> tuple[dict[int, Placement], set[int], set[int], str, list[str]]:
    descriptor = descriptor or {}
    if family == "original_layout":
        return dict(baseline), set(), set(), "original_baseline", ["original-inclusive baseline"]
    movable = movable_blocks(case)
    if family == "legacy_step7g_global_move":
        return _global_report_only_edit(case, baseline, frame, movable)
    if family == "vacancy_aware_local_insertion":
        return _local_feasible_shift(case, baseline, frame, movable)
    if family == "adjacent_region_reassignment":
        requested = max(3, int(descriptor.get("predicted_affected_blocks", 3)))
        return _regional_feasible_shift(case, baseline, frame, movable, requested=requested)
    if family == "mib_group_closure_macro":
        return _macro_feasible_shift(case, baseline, frame, movable)
    raise ValueError(f"unsupported Step7C-real-A family: {family}")


def evaluate_real_edit(candidate: RealPlacementEditCandidate) -> dict[str, Any]:
    hard_before = official_like_hard_summary(candidate.case, candidate.edited, candidate.frame)
    touched_regions = touched_region_count(
        candidate.edited, set(candidate.changed_blocks), candidate.frame
    )
    raw_fit_ratio = free_space_fit_ratio(
        candidate.edited,
        set(candidate.changed_blocks),
        candidate.frame,
    )
    # If the concrete real edit is already hard-feasible, the moved geometry has
    # demonstrably found space; cap the fit pressure so tight-but-legal one-block
    # nudges are not mislabeled regional solely because a coarse cell slack proxy is
    # pessimistic.  Infeasible/global report-only edits keep the raw pressure.
    fit_ratio = (
        min(raw_fit_ratio, 0.95) if hard_before["official_like_hard_feasible"] else raw_fit_ratio
    )
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.case.block_count,
        changed_block_count=len(candidate.changed_blocks),
        touched_region_count=touched_regions,
        macro_closure_size=candidate.predicted_macro_closure_size,
        min_region_slack=minimum_region_slack(candidate.edited, candidate.frame),
        free_space_fit_ratio=fit_ratio,
        hard_summary={"hard_feasible": hard_before["official_like_hard_feasible"]},
    )
    actual_class = str(prediction["predicted_locality_class"])
    repair = route_appropriate_real_repair_proxy(candidate, actual_class, hard_before)
    metrics = metric_delta_report(candidate.case, candidate.baseline, repair["after_route"])
    return {
        "case_id": candidate.case_id,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
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
        "changed_block_count": len(candidate.changed_blocks),
        "changed_block_fraction": len(candidate.changed_blocks)
        / max(candidate.case.block_count, 1),
        "affected_region_count": touched_regions,
        "macro_closure_size": candidate.predicted_macro_closure_size,
        "free_space_fit_ratio": fit_ratio,
        **{f"before_repair_{key}": value for key, value in hard_before.items()},
        **{f"after_route_{key}": value for key, value in repair["after_hard"].items()},
        **metrics,
        "runtime_proxy_ms": candidate.construction_ms + metrics["metric_eval_ms"],
        "failure_attribution": failure_attribution(
            candidate.descriptor_locality_class,
            actual_class,
            candidate.construction_status,
            hard_before,
            repair["after_hard"],
            repair["report_only"],
            metrics,
        ),
    }


def evaluate_real_edits(candidates: list[RealPlacementEditCandidate]) -> list[dict[str, Any]]:
    return [evaluate_real_edit(candidate) for candidate in candidates]


def route_appropriate_real_repair_proxy(
    candidate: RealPlacementEditCandidate,
    actual_class: str,
    hard_before: dict[str, Any],
) -> dict[str, Any]:
    if actual_class == "global":
        return {
            "route_lane": "global_report_only",
            "report_only": True,
            "after_route": candidate.edited,
            "after_hard": hard_before,
        }
    if actual_class == "local":
        if hard_before["official_like_hard_feasible"]:
            return {
                "route_lane": "bounded_local_noop_repair",
                "report_only": False,
                "after_route": candidate.edited,
                "after_hard": hard_before,
            }
        after = official_like_hard_summary(candidate.case, candidate.baseline, candidate.frame)
        return {
            "route_lane": "bounded_local_rollback",
            "report_only": False,
            "after_route": candidate.baseline,
            "after_hard": after,
        }
    if actual_class == "regional":
        return {
            "route_lane": "regional_report_lane",
            "report_only": False,
            "after_route": candidate.edited,
            "after_hard": hard_before,
        }
    return {
        "route_lane": "macro_report_lane",
        "report_only": False,
        "after_route": candidate.edited,
        "after_hard": hard_before,
    }


def official_like_hard_summary(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, Any]:
    ordered = _positions_list(placements, case.block_count)
    hard = summarize_hard_legality(case, ordered)
    frame_violations = sum(int(not frame.contains_box(box)) for box in ordered)
    return {
        "official_like_hard_feasible": bool(hard.is_feasible and frame_violations == 0),
        "official_legality_feasible": bool(hard.is_feasible),
        "overlap_violation_count": int(hard.overlap_violations),
        "area_tolerance_violation_count": int(hard.area_violations),
        "fixed_or_preplaced_violation_count": int(hard.dimension_violations),
        "boundary_or_frame_violation_count": int(frame_violations),
    }


def metric_delta_report(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    edited: dict[int, Placement],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    before_eval = _safe_evaluate(case, baseline)
    after_eval = _safe_evaluate(case, edited)
    before_bbox = bbox_area(baseline)
    after_bbox = bbox_area(edited)
    before_hpwl = hpwl_proxy(case, baseline)
    after_hpwl = hpwl_proxy(case, edited)
    before_soft = soft_delta_components(case, baseline)
    after_soft = soft_delta_components(case, edited)
    official_cost_delta = _nested_float(after_eval, "quality", "official_cost_raw") - _nested_float(
        before_eval, "quality", "official_cost_raw"
    )
    official_hpwl_delta = _nested_float(after_eval, "quality", "HPWLgap") - _nested_float(
        before_eval, "quality", "HPWLgap"
    )
    official_area_delta = _nested_float(after_eval, "quality", "Areagap_bbox") - _nested_float(
        before_eval, "quality", "Areagap_bbox"
    )
    official_violation_delta = _nested_float(
        after_eval, "quality", "Violationsrelative"
    ) - _nested_float(before_eval, "quality", "Violationsrelative")
    return {
        "hpwl_delta": after_hpwl - before_hpwl,
        "hpwl_delta_norm": (after_hpwl - before_hpwl) / max(abs(before_hpwl), 1e-9),
        "bbox_area_delta": after_bbox - before_bbox,
        "bbox_area_delta_norm": (after_bbox - before_bbox) / max(abs(before_bbox), 1e-9),
        "official_like_cost_delta": official_cost_delta,
        "official_like_hpwl_gap_delta": official_hpwl_delta,
        "official_like_area_gap_delta": official_area_delta,
        "official_like_violation_delta": official_violation_delta,
        "mib_group_boundary_soft_delta": {
            "mib_delta": after_soft["mib"] - before_soft["mib"],
            "grouping_delta": after_soft["grouping"] - before_soft["grouping"],
            "boundary_delta": after_soft["boundary"] - before_soft["boundary"],
            "total_delta": after_soft["total"] - before_soft["total"],
        },
        "official_eval_available": bool(
            before_eval.get("available") and after_eval.get("available")
        ),
        "official_before_quality": before_eval.get("quality", {}),
        "official_after_quality": after_eval.get("quality", {}),
        "metric_eval_ms": (time.perf_counter() - t0) * 1000.0,
    }


def soft_delta_components(case: FloorSetCase, placements: dict[int, Placement]) -> dict[str, int]:
    mib = int(mib_violation_count(case, placements))
    grouping = int(grouping_violation_count(case, placements))
    boundary = int(soft_boundary_violations(case, placements))
    return {
        "mib": mib,
        "grouping": grouping,
        "boundary": boundary,
        "total": mib + grouping + boundary,
    }


def confusion_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    stable = 0
    for row in rows:
        descriptor = str(row["descriptor_locality_class"])
        actual = str(row["actual_locality_class"])
        matrix[descriptor][actual] += 1
        stable += int(descriptor == actual)
    return {
        "descriptor_class_vs_actual_route_confusion": {
            key: dict(value) for key, value in sorted(matrix.items())
        },
        "descriptor_to_real_route_stability": stable / max(len(rows), 1),
        "stable_count": stable,
        "total": len(rows),
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
    descriptor_candidate_count: int,
    real_case_count: int,
) -> dict[str, Any]:
    actual_count = len(rows)
    class_counts = Counter(str(row["actual_locality_class"]) for row in rows)
    local_rows = [row for row in rows if row["actual_repair_mode"] == "bounded_repair_pareto"]
    invalid_local = [
        row for row in local_rows if not bool(row["before_repair_official_like_hard_feasible"])
    ]
    hard_feasible = [row for row in rows if bool(row["after_route_official_like_hard_feasible"])]
    safe = [row for row in rows if is_actual_safe_improvement(row)]
    regional_macro = [row for row in rows if row["actual_locality_class"] in {"regional", "macro"}]
    per_case_runtime: dict[str, float] = defaultdict(float)
    for row in rows:
        per_case_runtime[str(row["case_id"])] += float(row["runtime_proxy_ms"])
    return {
        "real_case_count": real_case_count,
        "descriptor_candidate_count": descriptor_candidate_count,
        "real_edit_candidate_count": actual_count,
        "real_route_count_by_class": dict(class_counts),
        "real_non_global_candidate_rate": _rate(
            actual_count - class_counts["global"], actual_count
        ),
        "invalid_local_attempt_rate": _rate(len(invalid_local), len(local_rows)),
        "official_like_hard_feasible_rate": _rate(len(hard_feasible), actual_count),
        "overlap_violation_count": sum(
            int(row["after_route_overlap_violation_count"]) for row in rows
        ),
        "area_tolerance_violation_count": sum(
            int(row["after_route_area_tolerance_violation_count"]) for row in rows
        ),
        "fixed_or_preplaced_violation_count": sum(
            int(row["after_route_fixed_or_preplaced_violation_count"]) for row in rows
        ),
        "boundary_or_frame_violation_count": sum(
            int(row["after_route_boundary_or_frame_violation_count"]) for row in rows
        ),
        "actual_safe_improvement_count": len(safe),
        "actual_safe_improvement_candidates": [_compact_identity(row) for row in safe],
        "regional_macro_preservation_count": len(regional_macro),
        "global_report_only_count": sum(int(bool(row["report_only"])) for row in rows),
        "runtime_proxy_per_case": dict(sorted(per_case_runtime.items())),
        "failure_attribution_counts": dict(Counter(row["failure_attribution"] for row in rows)),
        "invalid_or_non_improving_candidates": [
            _compact_identity(row) | {"failure_attribution": row["failure_attribution"]}
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hpwl_delta_distribution": _distribution(row["hpwl_delta"] for row in rows),
        "bbox_area_delta_distribution": _distribution(row["bbox_area_delta"] for row in rows),
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
        "mib_group_boundary_soft_delta": {
            key: _distribution(row["mib_group_boundary_soft_delta"][key] for row in rows)
            for key in ("mib_delta", "grouping_delta", "boundary_delta", "total_delta")
        },
        "official_eval_available_count": sum(int(row["official_eval_available"]) for row in rows),
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
            "original_included": any(row["family"] == "original_layout" for row in case_rows),
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
        "counts": dict(Counter(row["failure_attribution"] for row in rows)),
        "rows": [
            _compact_identity(row)
            | {
                "route_lane": row["route_lane"],
                "failure_attribution": row["failure_attribution"],
                "official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
                "official_like_cost_delta": row["official_like_cost_delta"],
                "hpwl_delta": row["hpwl_delta"],
                "bbox_area_delta": row["bbox_area_delta"],
            }
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def decision_for_step7c_real_a(
    feasibility: dict[str, Any],
    confusion: dict[str, Any],
    pareto: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    if metrics["official_eval_available_count"] < feasibility["real_edit_candidate_count"]:
        return "inconclusive_due_to_official_metric_gap"
    if confusion["descriptor_to_real_route_stability"] < 0.70:
        return "revisit_proxy_to_real_assumptions"
    if feasibility["real_non_global_candidate_rate"] <= 0.0:
        return "pivot_to_coarse_region_planner"
    if feasibility["invalid_local_attempt_rate"] > 0.05:
        return "build_route_specific_legalizers"
    if feasibility["official_like_hard_feasible_rate"] < 0.70:
        return "build_route_specific_legalizers"
    if pareto["original_inclusive_pareto_non_empty_count"] <= 0:
        return "inconclusive_due_to_official_metric_gap"
    improvement_count = int(feasibility["actual_safe_improvement_count"])
    non_original_count = max(
        feasibility["real_edit_candidate_count"] - feasibility["real_case_count"], 1
    )
    improvement_rate = improvement_count / non_original_count
    cost_dist = metrics["official_like_cost_delta_distribution"]
    if improvement_rate >= 0.20 and float(cost_dist["min"]) < 0.0:
        return "promote_to_step7c_multi_iteration_sidecar"
    if (
        feasibility["regional_macro_preservation_count"] > improvement_count
        and improvement_count == 0
    ):
        return "pivot_to_coarse_region_planner"
    return "refine_real_edit_generators"


def is_actual_safe_improvement(row: dict[str, Any]) -> bool:
    if not bool(row["after_route_official_like_hard_feasible"]):
        return False
    if row["actual_locality_class"] == "global":
        return False
    return (
        float(row["official_like_cost_delta"]) < -1e-9
        or float(row["hpwl_delta"]) < -1e-9
        or float(row["bbox_area_delta"]) < -1e-9
    )


def failure_attribution(
    descriptor_class: str,
    actual_class: str,
    construction_status: str,
    hard_before: dict[str, Any],
    hard_after: dict[str, Any],
    report_only: bool,
    metrics: dict[str, Any],
) -> str:
    if construction_status == "original_baseline":
        return "none"
    if descriptor_class != "global" and actual_class == "global":
        return "descriptor_collapsed_to_global_real_edit"
    if construction_status.startswith("no_feasible"):
        return construction_status
    if int(hard_after["overlap_violation_count"]) > 0:
        return "real_edit_overlap"
    if int(hard_after["area_tolerance_violation_count"]) > 0:
        return "real_edit_area_tolerance_violation"
    if int(hard_after["fixed_or_preplaced_violation_count"]) > 0:
        return "real_edit_fixed_or_preplaced_violation"
    if int(hard_after["boundary_or_frame_violation_count"]) > 0:
        return "real_edit_boundary_or_frame_violation"
    if report_only:
        return "global_report_only_not_repaired"
    if not bool(hard_before["official_like_hard_feasible"]):
        return "route_specific_legalizer_needed"
    if (
        float(metrics["official_like_cost_delta"]) >= -1e-9
        and float(metrics["hpwl_delta"]) >= -1e-9
        and float(metrics["bbox_area_delta"]) >= -1e-9
    ):
        return "real_edit_non_improving"
    return "none"


def movable_blocks(case: FloorSetCase) -> list[int]:
    return [
        idx
        for idx in range(case.block_count)
        if not bool(case.constraints[idx, ConstraintColumns.FIXED].item())
        and not bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    ]


def _global_report_only_edit(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    movable: list[int],
) -> tuple[dict[int, Placement], set[int], set[int], str, list[str]]:
    edited = dict(baseline)
    count = max(1, int(round(case.block_count * 0.80)))
    # Even report-only global probes must preserve fixed/preplaced exact handling;
    # they are broad disruptive candidates, not permission to corrupt hard anchors.
    chosen = movable[: min(count, len(movable))]
    anchor_x = frame.xmin - max(frame.width * 0.05, 1.0)
    anchor_y = frame.ymin - max(frame.height * 0.05, 1.0)
    for offset, idx in enumerate(chosen):
        _x, _y, w, h = edited[idx]
        edited[idx] = (anchor_x + (offset % 2) * 0.1, anchor_y + (offset // 2 % 2) * 0.1, w, h)
    return (
        edited,
        set(chosen),
        set(chosen),
        "constructed_global_report_only",
        ["intentionally broad report-only move; never routed to bounded-local repair"],
    )


def _local_feasible_shift(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    movable: list[int],
) -> tuple[dict[int, Placement], set[int], set[int], str, list[str]]:
    best = _search_feasible_shift(
        case, baseline, frame, [[idx] for idx in _rank_blocks_by_hpwl(case, baseline, movable)[:8]]
    )
    if best is None:
        return (
            dict(baseline),
            set(),
            set(),
            "no_feasible_local_real_edit",
            ["all tested single-block displacements violated official-like hard constraints"],
        )
    edited, changed, note = best
    return edited, changed, changed, "constructed_local_real_edit", [note]


def _regional_feasible_shift(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    movable: list[int],
    *,
    requested: int,
) -> tuple[dict[int, Placement], set[int], set[int], str, list[str]]:
    groups = _regional_groups(baseline, movable, requested=requested)
    best = _search_feasible_shift(case, baseline, frame, groups)
    if best is None:
        return (
            dict(baseline),
            set(),
            set(),
            "no_feasible_regional_real_edit",
            [
                "all tested region-neighborhood displacements violated "
                "official-like hard constraints"
            ],
        )
    edited, changed, note = best
    return edited, changed, changed, "constructed_regional_real_edit", [note]


def _macro_feasible_shift(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    movable: list[int],
) -> tuple[dict[int, Placement], set[int], set[int], str, list[str]]:
    groups = _constraint_groups(case, movable)
    best = _search_feasible_shift(case, baseline, frame, groups)
    if best is None:
        return (
            dict(baseline),
            set(),
            set().union(*(set(g) for g in groups)),
            "no_feasible_macro_real_edit",
            ["no MIB/group closure displacement passed official-like hard checks"],
        )
    edited, changed, note = best
    closure = set().union(
        *(set(group) for group in groups if set(changed).issubset(set(group)))
    ) or set(changed)
    return edited, changed, closure, "constructed_macro_real_edit", [note]


def _search_feasible_shift(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    block_groups: list[list[int]],
) -> tuple[dict[int, Placement], set[int], str] | None:
    best: tuple[tuple[float, float, float], dict[int, Placement], set[int], str] | None = None
    base_hpwl = hpwl_proxy(case, baseline)
    base_bbox = bbox_area(baseline)
    base_soft = soft_delta_components(case, baseline)["total"]
    for group in block_groups:
        group = [idx for idx in dict.fromkeys(group) if idx in baseline]
        if not group:
            continue
        for dx, dy in _candidate_vectors(case, baseline, group):
            edited = dict(baseline)
            for idx in group:
                x, y, w, h = edited[idx]
                edited[idx] = (x + dx, y + dy, w, h)
            hard = official_like_hard_summary(case, edited, frame)
            if not hard["official_like_hard_feasible"]:
                continue
            hpwl = hpwl_proxy(case, edited)
            area = bbox_area(edited)
            soft = soft_delta_components(case, edited)["total"]
            key = (hpwl - base_hpwl, area - base_bbox, float(soft - base_soft))
            note = f"shifted real block ids {group} by ({dx:.4g}, {dy:.4g})"
            if best is None or key < best[0]:
                best = (key, edited, set(group), note)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _candidate_vectors(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    group: list[int],
) -> list[tuple[float, float]]:
    cx, cy = _group_center(baseline, group)
    tx, ty = _weighted_target_center(case, baseline, group)
    dx = tx - cx
    dy = ty - cy
    norm = math.hypot(dx, dy)
    widths = [baseline[idx][2] for idx in group]
    heights = [baseline[idx][3] for idx in group]
    scale = max(min(min(widths + heights) * 0.20, norm * 0.25 if norm > 0 else 0.0), 0.05)
    primary = (0.0, 0.0) if norm <= 1e-9 else (dx / norm * scale, dy / norm * scale)
    grid_step = max(min(min(widths + heights) * 0.10, scale), 0.05)
    vectors = [primary]
    for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        d = math.hypot(sx, sy)
        vectors.append((sx / d * grid_step, sy / d * grid_step))
    # Try smaller versions first if the layout is tightly packed.
    out: list[tuple[float, float]] = []
    for factor in (1.0,):
        for vx, vy in vectors:
            if abs(vx) > 1e-12 or abs(vy) > 1e-12:
                out.append((vx * factor, vy * factor))
    return out


def _weighted_target_center(
    case: FloorSetCase, baseline: dict[int, Placement], group: list[int]
) -> tuple[float, float]:
    xs: list[tuple[float, float]] = []
    group_set = set(group)
    for src, dst, weight in case.b2b_edges.tolist():
        i = int(src)
        j = int(dst)
        w = abs(float(weight))
        if i in group_set and j in baseline and j not in group_set:
            xs.append((_center(baseline[j])[0], w))
            xs.append((_center(baseline[j])[1], w))
        elif j in group_set and i in baseline and i not in group_set:
            xs.append((_center(baseline[i])[0], w))
            xs.append((_center(baseline[i])[1], w))
    pxs: list[tuple[float, float]] = []
    pys: list[tuple[float, float]] = []
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        b = int(block_idx)
        p = int(pin_idx)
        if b not in group_set or p < 0 or p >= len(case.pins_pos):
            continue
        w = abs(float(weight))
        px, py = [float(v) for v in case.pins_pos[p].tolist()]
        pxs.append((px, w))
        pys.append((py, w))
    if pxs or pys:
        return _weighted_mean(pxs) if pxs else _group_center(baseline, group)[0], _weighted_mean(
            pys
        ) if pys else _group_center(baseline, group)[1]
    # Fallback to bbox center pull, which is deterministic and often compacting.
    bbox = _bbox_from_placements(baseline.values())
    if bbox is None:
        return _group_center(baseline, group)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    total_w = sum(w for _v, w in values)
    return sum(v * w for v, w in values) / max(total_w, 1e-9)


def _rank_blocks_by_hpwl(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    movable: list[int],
) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i in baseline and j in baseline:
            dist = _distance(_center(baseline[i]), _center(baseline[j])) * abs(float(weight))
            scores[i] += dist
            scores[j] += dist
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        p, b = int(pin_idx), int(block_idx)
        if b in baseline and 0 <= p < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[p].tolist()]
            scores[b] += _distance(_center(baseline[b]), (px, py)) * abs(float(weight))
    return sorted(movable, key=lambda idx: (-scores[idx], idx))


def _regional_groups(
    baseline: dict[int, Placement], movable: list[int], *, requested: int
) -> list[list[int]]:
    if not movable:
        return []
    count = max(3, min(requested, len(movable)))
    bbox = _bbox_from_placements(baseline.values())
    if bbox is None:
        return [movable[:count]]
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    quadrants = [
        [
            idx
            for idx in movable
            if _center(baseline[idx])[0] >= cx and _center(baseline[idx])[1] >= cy
        ],
        [
            idx
            for idx in movable
            if _center(baseline[idx])[0] < cx and _center(baseline[idx])[1] >= cy
        ],
        [
            idx
            for idx in movable
            if _center(baseline[idx])[0] >= cx and _center(baseline[idx])[1] < cy
        ],
        [
            idx
            for idx in movable
            if _center(baseline[idx])[0] < cx and _center(baseline[idx])[1] < cy
        ],
    ]
    groups = []
    for region in quadrants:
        ordered = sorted(
            region, key=lambda idx: (_center(baseline[idx])[0], _center(baseline[idx])[1], idx)
        )
        if ordered:
            groups.append(ordered[: min(count, len(ordered))])
    groups.append(sorted(movable)[:count])
    return groups


def _constraint_groups(case: FloorSetCase, movable: list[int]) -> list[list[int]]:
    movable_set = set(movable)
    groups: list[list[int]] = []
    for col in (ConstraintColumns.MIB, ConstraintColumns.CLUSTER):
        by_id: dict[int, list[int]] = defaultdict(list)
        for idx in movable:
            value = int(float(case.constraints[idx, col].item()))
            if value > 0:
                by_id[value].append(idx)
        groups.extend(sorted(members) for members in by_id.values() if members)
    groups.sort(key=lambda group: (-len(group), group[0]))
    if groups:
        return groups
    return [sorted(movable_set)[: max(2, min(4, len(movable_set)))]] if movable_set else []


def touched_region_count(
    placements: dict[int, Placement],
    changed: set[int],
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> int:
    regions = {_region_of(_center(placements[idx]), frame, rows=rows, cols=cols) for idx in changed}
    return len(regions)


def minimum_region_slack(placements: dict[int, Placement], frame: PuzzleFrame) -> float:
    rows = cols = 4
    cell_area = frame.area / (rows * cols)
    used: defaultdict[str, float] = defaultdict(float)
    for box in placements.values():
        used[_region_of(_center(box), frame, rows=rows, cols=cols)] += box[2] * box[3]
    return min((max(cell_area - value, 0.0) for value in used.values()), default=cell_area)


def free_space_fit_ratio(
    placements: dict[int, Placement], changed: set[int], frame: PuzzleFrame
) -> float:
    changed_area = sum(placements[idx][2] * placements[idx][3] for idx in changed)
    return changed_area / max(minimum_region_slack(placements, frame), 1e-9)


def changed_blocks(
    baseline: dict[int, Placement],
    edited: dict[int, Placement],
    *,
    eps: float = 1e-6,
) -> set[int]:
    return {
        idx
        for idx, before in baseline.items()
        if idx in edited
        and any(abs(a - b) > eps for a, b in zip(before, edited[idx], strict=False))
    }


def bbox_area(placements: dict[int, Placement]) -> float:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return 0.0
    return max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)


def _safe_evaluate(case: FloorSetCase, placements: dict[int, Placement]) -> dict[str, Any]:
    try:
        result = evaluate_positions(
            case, _positions_list(placements, case.block_count), runtime=1.0
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised only when official evaluator is unavailable.
        return {"available": False, "error": str(exc), "quality": {}}
    result["available"] = True
    return result


def _nested_float(payload: dict[str, Any], first: str, second: str) -> float:
    return float(payload.get(first, {}).get(second, 0.0))


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


def _region_of(point: tuple[float, float], frame: PuzzleFrame, *, rows: int, cols: int) -> str:
    x, y = point
    col = min(max(int((x - frame.xmin) / max(frame.width, 1e-9) * cols), 0), cols - 1)
    row = min(max(int((y - frame.ymin) / max(frame.height, 1e-9) * rows), 0), rows - 1)
    return f"r{row}_{col}"


def _route_lane(family: str) -> str:
    if family == "legacy_step7g_global_move":
        return "global_report_only"
    if family == "mib_group_closure_macro":
        return "macro_report_lane"
    if family == "adjacent_region_reassignment":
        return "regional_report_lane"
    return "bounded_local_candidate_lane"


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
        "family",
        "descriptor_locality_class",
        "actual_locality_class",
        "actual_repair_mode",
        "route_lane",
        "after_route_official_like_hard_feasible",
        "hpwl_delta",
        "bbox_area_delta",
        "official_like_cost_delta",
        "official_like_hpwl_gap_delta",
        "mib_group_boundary_soft_delta",
        "changed_block_fraction",
    ]
    return {key: row[key] for key in keys}


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "candidate_id",
        "family",
        "actual_locality_class",
        "after_route_official_like_hard_feasible",
        "hpwl_delta",
        "hpwl_delta_norm",
        "bbox_area_delta",
        "bbox_area_delta_norm",
        "official_like_cost_delta",
        "official_like_hpwl_gap_delta",
        "official_like_area_gap_delta",
        "official_like_violation_delta",
        "mib_group_boundary_soft_delta",
    ]
    return {key: row[key] for key in keys}


def _compact_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "family": row["family"],
        "descriptor_locality_class": row["descriptor_locality_class"],
        "actual_locality_class": row["actual_locality_class"],
    }


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
