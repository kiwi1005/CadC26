from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from puzzleplace.alternatives.pareto_slack_fit_edits import (
    BASELINE_STEP7C_REAL_B,
    DEFAULT_EPSILON,
    dominance_report,
    feasibility_and_improvement_report,
    pareto_front_report,
)

STEP7C_REAL_E_BASELINE = {
    "source_candidate_count": 354,
    "official_like_cost_improving_count": 16,
    "official_like_cost_improvement_density": 0.046242774566473986,
    "hpwl_to_official_like_conversion_rate": 0.1797752808988764,
    "dominated_by_original_count": 258,
}

DEFAULT_RANK_BUDGET = 12
RANK_BUDGET_SENSITIVITY = (1, 2, 4, 6, 8, 10, 12)


def preselect_dominance_slack_candidates(
    rows: list[dict[str, Any]], *, rank_budget: int = DEFAULT_RANK_BUDGET
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_id"])].append(row)

    retained_ids: set[str] = set()
    reasons: dict[str, str] = {}
    feature_rows: list[dict[str, Any]] = []

    for case_id, case_rows in sorted(by_case.items()):
        for row in case_rows:
            cid = str(row["candidate_id"])
            if row["strategy"] == "original_layout":
                retained_ids.add(cid)
                reasons[cid] = "retain_original_anchor"
            elif row["strategy"] == "legacy_step7g_global_report":
                retained_ids.add(cid)
                reasons[cid] = "retain_global_report_only"

        eligible = [row for row in case_rows if _is_preselection_eligible(row)]
        ranked = sorted(eligible, key=_preselection_rank_key)
        for rank, row in enumerate(ranked):
            cid = str(row["candidate_id"])
            feature_rows.append(_feature_row(row, case_id=case_id, rank=rank, eligible=True))
            if rank < rank_budget:
                retained_ids.add(cid)
                reasons[cid] = "retain_proxy_pareto_rank_budget"
            else:
                reasons[cid] = "filtered_rank_budget_exceeded"
        for row in case_rows:
            cid = str(row["candidate_id"])
            if cid in reasons:
                continue
            reasons[cid] = _filter_reason(row)
            feature_rows.append(_feature_row(row, case_id=case_id, rank=None, eligible=False))

    retained = [
        _copy_with_filter(row, reasons[str(row["candidate_id"])])
        for row in rows
        if str(row["candidate_id"]) in retained_ids
    ]
    filtered = [
        _copy_with_filter(row, reasons[str(row["candidate_id"])])
        for row in rows
        if str(row["candidate_id"]) not in retained_ids
    ]
    report = filter_report(rows, retained, filtered, reasons, feature_rows, rank_budget=rank_budget)
    return retained, report


def filter_report(
    source_rows: list[dict[str, Any]],
    retained_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    reasons: dict[str, str],
    feature_rows: list[dict[str, Any]],
    *,
    rank_budget: int,
) -> dict[str, Any]:
    e_winners = [row for row in source_rows if row.get("official_like_cost_improving")]
    retained_ids = {row["candidate_id"] for row in retained_rows}
    preserved = [row for row in e_winners if row["candidate_id"] in retained_ids]
    return {
        "rank_budget": rank_budget,
        "real_case_count": len({int(row["case_id"]) for row in source_rows}),
        "source_candidate_count_from_E": len(source_rows),
        "generated_candidate_count": len(retained_rows),
        "filtered_candidate_count": len(filtered_rows),
        "retained_candidate_count": len(retained_rows),
        "retained_fraction": len(retained_rows) / max(len(source_rows), 1),
        "filter_reason_counts": dict(Counter(reasons.values())),
        "E_official_like_winner_count": len(e_winners),
        "E_winner_preservation_count": len(preserved),
        "E_winner_preservation_rate": len(preserved) / max(len(e_winners), 1),
        "retained_strategy_counts": dict(Counter(str(row["strategy"]) for row in retained_rows)),
        "filtered_strategy_counts": dict(Counter(str(row["strategy"]) for row in filtered_rows)),
        "preselection_feature_rows": sorted(
            feature_rows,
            key=lambda row: (
                row["case_id"],
                row["rank"] is None,
                row.get("rank") or 0,
                row["candidate_id"],
            ),
        ),
        "filtered_candidates": [
            _compact(row) | {"filter_reason": row["filter_reason"]} for row in filtered_rows
        ],
    }


def objective_vector_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "candidate_id": row["candidate_id"],
            "strategy": row["strategy"],
            "actual_locality_class": row["actual_locality_class"],
            "filter_reason": row.get("filter_reason", "unlabeled"),
            "objective_vector": row["objective_vector"],
            "official_like_cost_improving": row["official_like_cost_improving"],
            "hpwl_improving": row["hpwl_improving"],
        }
        for row in rows
    ]


def retained_measurement_report(
    retained_rows: list[dict[str, Any]], filter_summary: dict[str, Any]
) -> dict[str, Any]:
    pareto = pareto_front_report(retained_rows)
    dominance = dominance_report(retained_rows)
    feasibility = feasibility_and_improvement_report(retained_rows, pareto, dominance)
    source_count = int(filter_summary["source_candidate_count_from_E"])
    dominated = int(feasibility["candidates_dominated_by_original_count"])
    feasibility.update(
        {
            "source_candidate_count_from_E": source_count,
            "generated_candidate_count": len(retained_rows),
            "filtered_candidate_count": int(filter_summary["filtered_candidate_count"]),
            "retained_candidate_count": len(retained_rows),
            "retained_fraction": float(filter_summary["retained_fraction"]),
            "filter_reason_counts": filter_summary["filter_reason_counts"],
            "E_winner_preservation_count": filter_summary["E_winner_preservation_count"],
            "E_winner_preservation_rate": filter_summary["E_winner_preservation_rate"],
            "dominated_by_original_count": dominated,
            "dominated_by_original_rate": dominated / max(len(retained_rows), 1),
            "hpwl_to_official_like_conversion_rate": feasibility[
                "hpwl_gain_to_official_like_conversion_rate"
            ],
        }
    )
    return feasibility


def rank_budget_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sections = []
    for budget in RANK_BUDGET_SENSITIVITY:
        retained, report = preselect_dominance_slack_candidates(rows, rank_budget=budget)
        pareto = pareto_front_report(retained)
        dominance = dominance_report(retained)
        feasibility = feasibility_and_improvement_report(retained, pareto, dominance)
        sections.append(
            {
                "rank_budget": budget,
                "retained_candidate_count": len(retained),
                "official_like_cost_improving_count": feasibility[
                    "official_like_cost_improving_count"
                ],
                "official_like_cost_improvement_density": feasibility[
                    "official_like_cost_improvement_density"
                ],
                "hpwl_to_official_like_conversion_rate": feasibility[
                    "hpwl_gain_to_official_like_conversion_rate"
                ],
                "dominated_by_original_count": feasibility[
                    "candidates_dominated_by_original_count"
                ],
                "E_winner_preservation_count": report["E_winner_preservation_count"],
                "E_winner_preservation_rate": report["E_winner_preservation_rate"],
                "original_inclusive_pareto_non_empty_count": pareto[
                    "original_inclusive_pareto_non_empty_count"
                ],
            }
        )
    return {
        "budget_source": "per-case top-k rank over proxy Pareto-eligible local slack candidates",
        "sensitivity_to_rank_budget_or_top_k": sections,
    }


def decision_for_step7c_real_f(feasibility: dict[str, Any], route: dict[str, Any]) -> str:
    density = float(feasibility["official_like_cost_improvement_density"])
    conversion = float(feasibility["hpwl_to_official_like_conversion_rate"])
    official_count = int(feasibility["official_like_cost_improving_count"])
    dominated_rate = float(feasibility["dominated_by_original_rate"])
    preserved = float(feasibility["E_winner_preservation_rate"])
    if route["invalid_local_attempt_rate"] > 0.05:
        return "combine_preselection_with_route_specific_legalizer"
    if float(feasibility["official_like_hard_feasible_rate"]) < 0.70:
        return "combine_preselection_with_route_specific_legalizer"
    if int(feasibility["original_inclusive_pareto_non_empty_count"]) < int(
        feasibility["real_case_count"]
    ):
        return "inconclusive_due_to_preselection_quality"
    if (
        density > BASELINE_STEP7C_REAL_B["official_like_cost_improvement_density"]
        and conversion >= 0.25
        and dominated_rate
        < STEP7C_REAL_E_BASELINE["dominated_by_original_count"]
        / STEP7C_REAL_E_BASELINE["source_candidate_count"]
        and official_count >= 12
        and preserved >= 0.75
    ):
        return "promote_to_step7c_local_lane_iteration"
    if density > STEP7C_REAL_E_BASELINE["official_like_cost_improvement_density"]:
        return "refine_dominance_preselection"
    if official_count > 0 and dominated_rate > 0.50:
        return "revisit_local_slack_lane_ceiling"
    return "inconclusive_due_to_preselection_quality"


def _is_preselection_eligible(row: dict[str, Any]) -> bool:
    return (
        row["strategy"] not in {"original_layout", "legacy_step7g_global_report"}
        and row["actual_locality_class"] == "local"
        and bool(row["after_route_official_like_hard_feasible"])
        and int(row["changed_block_count"]) == 1
        and float(row.get("proxy_hpwl_delta", row["hpwl_delta"])) < -DEFAULT_EPSILON
        and float(row.get("proxy_bbox_delta", row["bbox_area_delta"])) <= DEFAULT_EPSILON
        and float(row.get("proxy_soft_delta", row["mib_group_boundary_soft_delta"]["total_delta"]))
        <= DEFAULT_EPSILON
    )


def _preselection_rank_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row.get("proxy_hpwl_delta", row["hpwl_delta"])),
        float(row.get("displacement_magnitude", 0.0)),
        float(row.get("free_space_fit_ratio", 0.0)),
        str(row["candidate_id"]),
    )


def _filter_reason(row: dict[str, Any]) -> str:
    if row["strategy"] in {"original_layout", "legacy_step7g_global_report"}:
        return "retained_anchor_or_report_only"
    if row["actual_locality_class"] != "local":
        return "filtered_route_not_local"
    if not bool(row["after_route_official_like_hard_feasible"]):
        return "filtered_hard_infeasible"
    if int(row["changed_block_count"]) != 1:
        return "filtered_not_one_block"
    if float(row.get("proxy_hpwl_delta", row["hpwl_delta"])) >= -DEFAULT_EPSILON:
        return "filtered_proxy_not_hpwl_improving"
    if float(row.get("proxy_bbox_delta", row["bbox_area_delta"])) > DEFAULT_EPSILON:
        return "filtered_proxy_bbox_regression"
    if (
        float(row.get("proxy_soft_delta", row["mib_group_boundary_soft_delta"]["total_delta"]))
        > DEFAULT_EPSILON
    ):
        return "filtered_proxy_soft_risk"
    return "filtered_unknown"


def _copy_with_filter(row: dict[str, Any], reason: str) -> dict[str, Any]:
    copied = dict(row)
    copied["filter_reason"] = reason
    return copied


def _feature_row(
    row: dict[str, Any], *, case_id: int, rank: int | None, eligible: bool
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "candidate_id": row["candidate_id"],
        "strategy": row["strategy"],
        "eligible": eligible,
        "rank": rank,
        "proxy_hpwl_delta": row.get("proxy_hpwl_delta"),
        "proxy_bbox_delta": row.get("proxy_bbox_delta"),
        "proxy_soft_delta": row.get("proxy_soft_delta"),
        "displacement_magnitude": row.get("displacement_magnitude"),
        "free_space_fit_ratio": row.get("free_space_fit_ratio"),
        "official_like_cost_improving": row.get("official_like_cost_improving", False),
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "strategy": row["strategy"],
        "actual_locality_class": row["actual_locality_class"],
        "official_like_cost_delta": row["official_like_cost_delta"],
        "hpwl_delta": row["hpwl_delta"],
        "bbox_area_delta": row["bbox_area_delta"],
        "soft_constraint_delta": row["mib_group_boundary_soft_delta"]["total_delta"],
        "changed_block_count": row["changed_block_count"],
        "official_like_cost_improving": row["official_like_cost_improving"],
    }
