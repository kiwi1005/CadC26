from __future__ import annotations

from collections import Counter
from typing import Any, Literal

LocalityClass = Literal["local", "regional", "macro", "global"]
RoutingDecision = Literal[
    "bounded_repair_pareto",
    "region_repair_or_planner",
    "macro_legalizer",
    "global_route_not_local_selector",
]


def predict_move_locality(
    *,
    case_id: int,
    block_count: int,
    changed_block_count: int,
    touched_region_count: int,
    macro_closure_size: int,
    min_region_slack: float,
    free_space_fit_ratio: float,
    hard_summary: dict[str, Any],
) -> dict[str, Any]:
    changed_fraction = changed_block_count / max(block_count, 1)
    macro_fraction = macro_closure_size / max(block_count, 1)
    if changed_fraction >= 0.50 or touched_region_count >= 8:
        locality_class: LocalityClass = "global"
    elif macro_fraction >= 0.25 and macro_closure_size > changed_block_count:
        locality_class = "macro"
    elif changed_fraction >= 0.15 or touched_region_count >= 3 or free_space_fit_ratio > 1.0:
        locality_class = "regional"
    else:
        locality_class = "local"
    return {
        "case_id": case_id,
        "predicted_affected_blocks": changed_block_count,
        "predicted_affected_block_fraction": changed_fraction,
        "predicted_affected_regions": touched_region_count,
        "predicted_macro_closure_size": macro_closure_size,
        "predicted_region_slack": min_region_slack,
        "predicted_free_space_fit": free_space_fit_ratio <= 1.0,
        "predicted_free_space_fit_ratio": free_space_fit_ratio,
        "predicted_repair_mode": route_for_class(locality_class),
        "predicted_locality_class": locality_class,
        "hard_invalid_before_repair": not bool(hard_summary.get("hard_feasible", False)),
    }


def route_for_class(locality_class: LocalityClass) -> RoutingDecision:
    if locality_class == "local":
        return "bounded_repair_pareto"
    if locality_class == "regional":
        return "region_repair_or_planner"
    if locality_class == "macro":
        return "macro_legalizer"
    return "global_route_not_local_selector"


def actual_locality_from_step7f(rows: list[dict[str, Any]]) -> LocalityClass:
    labels = actual_weak_labels_from_step7f(rows)
    moved_fraction = float(labels["actual_moved_block_fraction"])
    affected_regions = int(labels["actual_affected_region_count"])
    macro_violations = int(labels["actual_mib_group_violation_after"])
    radius_exceeded = bool(labels["actual_radius_exceeded"])
    if moved_fraction >= 0.50 or affected_regions >= 8 or radius_exceeded:
        return "global"
    if macro_violations > 0:
        return "macro"
    if moved_fraction >= 0.15 or affected_regions >= 3:
        return "regional"
    return "local"


def actual_weak_labels_from_step7f(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = next(
        (row for row in rows if row["repair_mode"] == "current_repair_baseline"),
        rows[0],
    )
    radius_exceeded = any(bool(row.get("repair_radius_exceeded")) for row in rows)
    hard_feasible_after = bool(current.get("hard_feasible_after", False))
    frame_protrusion = float(current.get("frame_protrusion_after", 0.0))
    overlap_count = int(current.get("overlap_count_after", 0))
    mib_group_violations = int(current.get("MIB_group_violation_after", 0))
    if hard_feasible_after and not radius_exceeded:
        attribution = "safe_or_feasible"
    elif radius_exceeded:
        attribution = "radius_global_cascade"
    elif mib_group_violations > 0:
        attribution = "MIB_group_violation"
    elif frame_protrusion > 0.0:
        attribution = "frame_protrusion"
    elif overlap_count > 0:
        attribution = "overlap_residual"
    else:
        attribution = "unknown_hard_failure"
    return {
        "actual_moved_block_fraction": float(current.get("moved_block_fraction", 0.0)),
        "actual_affected_region_count": int(current.get("affected_region_count", 0)),
        "actual_radius_exceeded": radius_exceeded,
        "actual_hard_feasible_after": hard_feasible_after,
        "actual_failure_attribution": attribution,
        "actual_mib_group_violation_after": mib_group_violations,
        "actual_frame_protrusion_after": frame_protrusion,
        "actual_overlap_count_after": overlap_count,
    }


def calibration_report(
    predictions: list[dict[str, Any]],
    step7f_rows_by_case: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for prediction in predictions:
        case_id = int(prediction["case_id"])
        step7f_rows = step7f_rows_by_case[case_id]
        actual = actual_locality_from_step7f(step7f_rows)
        weak_labels = actual_weak_labels_from_step7f(step7f_rows)
        predicted = str(prediction["predicted_locality_class"])
        relation = calibration_relation(predicted, actual)
        counts[relation] += 1
        rows.append(
            {
                "case_id": case_id,
                "predicted_locality_class": predicted,
                "actual_locality_class": actual,
                "relation": relation,
                "routing_decision": prediction["predicted_repair_mode"],
                **weak_labels,
            }
        )
    return {
        "rows": rows,
        "counts": {
            "correct_local": counts["correct_local"],
            "correct_regional": counts["correct_regional"],
            "correct_macro": counts["correct_macro"],
            "correct_global": counts["correct_global"],
            "under_predicted_globality": counts["under_predicted_globality"],
            "over_predicted_globality": counts["over_predicted_globality"],
        },
        "accuracy": sum(1 for row in rows if row["relation"].startswith("correct_"))
        / max(len(rows), 1),
    }


def calibration_relation(predicted: str, actual: str) -> str:
    if predicted == actual:
        return f"correct_{actual}"
    rank = {"local": 0, "regional": 1, "macro": 2, "global": 3}
    if rank[predicted] < rank[actual]:
        return "under_predicted_globality"
    return "over_predicted_globality"


def routing_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, Counter[str]] = {}
    for row in predictions:
        locality_class = str(row["predicted_locality_class"])
        source = str(row.get("source_move_type", "unknown"))
        by_class.setdefault(locality_class, Counter())[source] += 1
    return {
        "class_counts": dict(Counter(row["predicted_locality_class"] for row in predictions)),
        "route_counts": dict(Counter(row["predicted_repair_mode"] for row in predictions)),
        "local_selector_candidate_count": sum(
            int(row["predicted_repair_mode"] == "bounded_repair_pareto")
            for row in predictions
        ),
        "nonlocal_route_count": sum(
            int(row["predicted_repair_mode"] != "bounded_repair_pareto")
            for row in predictions
        ),
        "source_move_type_counts_by_class": {
            locality_class: dict(counter) for locality_class, counter in by_class.items()
        },
    }


def routing_quality_report(
    predictions: list[dict[str, Any]],
    step7f_rows_by_case: dict[int, list[dict[str, Any]]],
    pareto_selection: dict[str, Any],
) -> dict[str, Any]:
    """Report whether locality routing preserves useful alternatives.

    Step7G is classification plus routing, not rejection.  The "after routing"
    counts therefore keep non-local alternatives in the report/non-local branch
    while separately showing what a local-only selector would see.
    """

    case_ids = [int(row["case_id"]) for row in predictions]
    weak_by_case = {
        case_id: actual_weak_labels_from_step7f(step7f_rows_by_case[case_id])
        for case_id in case_ids
    }
    invalid_before = [
        case_id
        for case_id in case_ids
        if _invalid_local_attempt(weak_by_case[case_id])
        or bool(_prediction_for_case(predictions, case_id).get("hard_invalid_before_repair"))
    ]
    local_cases = [
        int(row["case_id"])
        for row in predictions
        if row["predicted_repair_mode"] == "bounded_repair_pareto"
    ]
    invalid_after_local = [
        case_id for case_id in local_cases if _invalid_local_attempt(weak_by_case[case_id])
    ]
    useful_before = sorted(
        case_id for case_id in case_ids if _has_safe_improvement(pareto_selection, case_id)
    )
    useful_local_after = sorted(case_id for case_id in local_cases if case_id in useful_before)
    useful_preserved_after = useful_before[:]
    useful_nonlocal = [
        case_id
        for case_id in useful_before
        if _prediction_for_case(predictions, case_id)["predicted_repair_mode"]
        != "bounded_repair_pareto"
    ]
    front_non_empty = sorted(
        case_id for case_id in case_ids if _front_entries(pareto_selection, case_id)
    )
    route_diversity = routing_summary(predictions)
    return {
        "candidate_count": len(predictions),
        "hard_invalid_before_repair_count": sum(
            int(bool(row.get("hard_invalid_before_repair"))) for row in predictions
        ),
        "hard_invalid_before_repair_rate": _rate(
            sum(int(bool(row.get("hard_invalid_before_repair"))) for row in predictions),
            len(predictions),
        ),
        "invalid_local_repair_attempt_count_before_routing": len(invalid_before),
        "invalid_local_repair_attempt_rate_before_routing": _rate(
            len(invalid_before), len(predictions)
        ),
        "local_repair_attempt_count_after_routing": len(local_cases),
        "invalid_local_repair_attempt_count_after_routing": len(invalid_after_local),
        "invalid_local_repair_attempt_rate_after_routing": _rate(
            len(invalid_after_local), len(local_cases)
        ),
        "safe_improvement_count_before_routing": len(useful_before),
        "safe_improvement_cases_before_routing": useful_before,
        "safe_improvement_count_after_routing_preserved": len(useful_preserved_after),
        "safe_improvement_cases_after_routing_preserved": useful_preserved_after,
        "safe_improvement_count_after_routing_local_selector_only": len(useful_local_after),
        "safe_improvement_cases_after_routing_local_selector_only": useful_local_after,
        "useful_improvements_requiring_nonlocal_followup": useful_nonlocal,
        "useful_improvements_lost_by_over_aggressive_prediction": [],
        "if_nonlocal_routes_were_discarded_useful_improvements_lost": useful_nonlocal,
        "pareto_front_non_empty_count_before_routing": len(front_non_empty),
        "pareto_front_non_empty_cases_before_routing": front_non_empty,
        "pareto_front_non_empty_count_after_routing_preserved": len(front_non_empty),
        "pareto_front_non_empty_cases_after_routing_preserved": front_non_empty,
        "pareto_front_non_empty_count_if_local_only": len(
            [case_id for case_id in front_non_empty if case_id in local_cases]
        ),
        "move_diversity_by_route_class": route_diversity[
            "source_move_type_counts_by_class"
        ],
        "original_layout_preserved_cases": sorted(
            case_id
            for case_id in case_ids
            if any(
                row.get("repair_mode") == "rollback_to_original"
                for row in _front_entries(pareto_selection, case_id)
            )
        ),
    }


def _prediction_for_case(predictions: list[dict[str, Any]], case_id: int) -> dict[str, Any]:
    return next(row for row in predictions if int(row["case_id"]) == case_id)


def _invalid_local_attempt(labels: dict[str, Any]) -> bool:
    return (not bool(labels["actual_hard_feasible_after"])) or bool(
        labels["actual_radius_exceeded"]
    )


def _has_safe_improvement(pareto_selection: dict[str, Any], case_id: int) -> bool:
    for row in _front_entries(pareto_selection, case_id):
        if row.get("repair_mode") == "rollback_to_original":
            continue
        if not bool(row.get("hard_feasible_after")):
            continue
        if bool(row.get("repair_radius_exceeded")):
            continue
        if (
            float(row.get("hpwl_delta_norm", 0.0)) < 0.0
            or float(row.get("bbox_delta_norm", 0.0)) < 0.0
            or float(row.get("boundary_delta", 0.0)) > 0.0
        ):
            return True
    return False


def _front_entries(pareto_selection: dict[str, Any], case_id: int) -> list[dict[str, Any]]:
    section = pareto_selection.get(str(case_id), {})
    front = section.get("front", [])
    if isinstance(front, list):
        return [row for row in front if isinstance(row, dict)]
    return []


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
