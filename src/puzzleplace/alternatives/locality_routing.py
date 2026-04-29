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
    }
