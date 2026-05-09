from __future__ import annotations

from puzzleplace.alternatives.dominance_slack_preselection import (
    decision_for_step7c_real_f,
    preselect_dominance_slack_candidates,
    rank_budget_sensitivity,
    retained_measurement_report,
)
from puzzleplace.alternatives.pareto_slack_fit_edits import route_report


def _row(
    cid: str,
    *,
    strategy: str = "slack_fit_insertion_expanded",
    hpwl: float = -0.1,
    bbox: float = 0.0,
    soft: int = 0,
    official: float = -0.01,
    winner: bool = True,
    route: str = "local",
) -> dict[str, object]:
    feasible = route != "global"
    return {
        "case_id": 0,
        "candidate_id": cid,
        "strategy": strategy,
        "descriptor_locality_class": route,
        "actual_locality_class": route,
        "actual_repair_mode": "bounded_repair_pareto"
        if route == "local"
        else "global_route_not_local_selector",
        "route_lane": "bounded_local_noop_repair" if route == "local" else "global_report_only",
        "report_only": route == "global",
        "after_route_official_like_hard_feasible": feasible,
        "before_repair_official_like_hard_feasible": feasible,
        "changed_block_count": 1
        if strategy not in {"original_layout", "legacy_step7g_global_report"}
        else 0,
        "changed_block_fraction": 0.1,
        "hpwl_delta": hpwl,
        "bbox_area_delta": bbox,
        "official_like_cost_delta": official,
        "official_like_cost_improving": winner,
        "hpwl_improving": hpwl < 0.0,
        "safe_improvement": winner,
        "proxy_hpwl_delta": hpwl,
        "proxy_bbox_delta": bbox,
        "proxy_soft_delta": float(soft),
        "displacement_magnitude": abs(hpwl),
        "free_space_fit_ratio": 0.5,
        "runtime_proxy_ms": 1.0,
        "mib_group_boundary_soft_delta": {
            "mib_delta": 0,
            "grouping_delta": 0,
            "boundary_delta": soft,
            "total_delta": soft,
        },
        "objective_vector": {
            "feasibility_rank": 0 if feasible else 1,
            "route_rank": 0 if route != "global" else 1,
            "official_like_cost_delta": official,
            "hpwl_delta": hpwl,
            "bbox_area_delta": bbox,
            "soft_constraint_delta": float(soft),
            "changed_block_fraction": 0.1,
        },
    }


def _rows() -> list[dict[str, object]]:
    return [
        _row(
            "case000:original_layout",
            strategy="original_layout",
            hpwl=0.0,
            official=0.0,
            winner=False,
        ),
        _row(
            "case000:legacy_step7g_global_report",
            strategy="legacy_step7g_global_report",
            route="global",
            hpwl=5.0,
            bbox=10.0,
            official=9.0,
            winner=False,
        ),
        _row("case000:good0", hpwl=-0.5, official=-0.05, winner=True),
        _row("case000:good1", hpwl=-0.4, official=-0.04, winner=True),
        _row("case000:budgeted", hpwl=-0.3, official=-0.03, winner=True),
        _row("case000:no_hpwl", hpwl=0.2, official=0.1, winner=False),
        _row("case000:bbox_bad", hpwl=-0.2, bbox=1.0, official=0.2, winner=False),
        _row("case000:soft_bad", hpwl=-0.2, soft=1, official=0.2, winner=False),
    ]


def test_preselection_retains_original_global_and_ranked_proxy_candidates() -> None:
    retained, report = preselect_dominance_slack_candidates(_rows(), rank_budget=2)  # type: ignore[arg-type]
    ids = {row["candidate_id"] for row in retained}
    assert "case000:original_layout" in ids
    assert "case000:legacy_step7g_global_report" in ids
    assert "case000:good0" in ids
    assert "case000:good1" in ids
    assert "case000:budgeted" not in ids
    assert report["filter_reason_counts"]["filtered_rank_budget_exceeded"] == 1
    assert report["E_winner_preservation_count"] == 2


def test_preselection_reports_measurements_and_sensitivity() -> None:
    retained, report = preselect_dominance_slack_candidates(_rows(), rank_budget=2)  # type: ignore[arg-type]
    feasibility = retained_measurement_report(retained, report)  # type: ignore[arg-type]
    route = route_report(retained)  # type: ignore[arg-type]
    sensitivity = rank_budget_sensitivity(_rows())  # type: ignore[arg-type]
    assert feasibility["source_candidate_count_from_E"] == 8
    assert feasibility["generated_candidate_count"] == 4
    assert feasibility["filtered_candidate_count"] == 4
    assert feasibility["official_like_cost_improving_count"] == 2
    assert route["global_report_only_count"] == 1
    assert sensitivity["sensitivity_to_rank_budget_or_top_k"]


def test_decision_promotes_when_density_conversion_and_preservation_are_strong() -> None:
    retained, report = preselect_dominance_slack_candidates(_rows(), rank_budget=2)  # type: ignore[arg-type]
    feasibility = retained_measurement_report(retained, report)  # type: ignore[arg-type]
    feasibility["real_case_count"] = 1
    feasibility["official_like_cost_improving_count"] = 12
    feasibility["E_winner_preservation_rate"] = 0.8
    route = route_report(retained)  # type: ignore[arg-type]
    assert (
        decision_for_step7c_real_f(feasibility, route) == "promote_to_step7c_local_lane_iteration"
    )
