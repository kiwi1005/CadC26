from __future__ import annotations

from typing import Any

from puzzleplace.alternatives.pareto_slack_fit_edits import objective_vector
from puzzleplace.search.local_lane_iteration import (
    decision_for_local_iter0,
    select_current_pareto_local_edit,
    skip_reason_for_iteration,
)


def _row(
    cid: str,
    *,
    strategy: str = "nearby_slack_slot_insertion",
    route: str = "local",
    official_delta: float = -0.01,
    hpwl_delta: float = -0.1,
    bbox_delta: float = 0.0,
    soft_delta: int = 0,
    feasible: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": 0,
        "iteration": 0,
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
        "official_like_cost_delta": official_delta,
        "hpwl_delta": hpwl_delta,
        "bbox_area_delta": bbox_delta,
        "mib_group_boundary_soft_delta": {
            "mib_delta": 0,
            "grouping_delta": 0,
            "boundary_delta": soft_delta,
            "total_delta": soft_delta,
        },
        "official_like_cost_improving": official_delta < 0.0 and feasible,
        "hpwl_improving": hpwl_delta < 0.0,
        "safe_improvement": official_delta < 0.0 and feasible,
        "runtime_proxy_ms": 1.0,
    }
    row["objective_vector"] = objective_vector(row)
    return row


def test_select_current_pareto_local_edit_ignores_global_and_picks_best_official_delta() -> None:
    rows = [
        _row("original", strategy="original_layout", official_delta=0.0, hpwl_delta=0.0),
        _row("global", strategy="legacy_step7g_global_report", route="global", official_delta=-9.0),
        _row("good", official_delta=-0.2, hpwl_delta=-0.2),
        _row("better", official_delta=-0.3, hpwl_delta=-0.1),
    ]
    selected = select_current_pareto_local_edit(rows)
    assert selected is not None
    assert selected["candidate_id"] == "better"


def test_skip_reason_distinguishes_preselection_starvation_from_no_improvement() -> None:
    generated = [_row("generated")]
    retained = [_row("original", strategy="original_layout", official_delta=0.0, hpwl_delta=0.0)]
    assert (
        skip_reason_for_iteration(generated, retained)
        == "candidate_starvation_no_preselection_retained"
    )
    retained.append(_row("non_improving", official_delta=0.1, hpwl_delta=-0.1))
    assert (
        skip_reason_for_iteration(generated, retained)
        == "no_official_like_improving_local_candidate"
    )


def test_decision_refines_when_only_two_of_eight_cases_improve() -> None:
    reports = {
        "candidate_report": {
            "real_case_count": 8,
            "selected_edit_count": 2,
            "candidate_starvation_count": 0,
        },
        "feasibility_report": {
            "invalid_local_attempt_rate": 0.0,
            "sequential_feasibility_collapse_count": 0,
        },
        "metric_report": {
            "cumulative_official_like_cost_delta_by_case": {
                "19": 0.0,
                "24": -1e-6,
                "25": -1e-6,
                "51": 0.0,
                "76": 0.0,
                "79": 0.0,
                "91": 0.0,
                "99": 0.0,
            }
        },
        "pareto_report": {"original_inclusive_pareto_non_empty_count_by_iteration": {"0": 8}},
    }
    assert decision_for_local_iter0(reports) == "refine_sequential_local_preselection"


def test_decision_promotes_when_half_the_cases_improve_without_collapse() -> None:
    reports = {
        "candidate_report": {
            "real_case_count": 8,
            "selected_edit_count": 4,
            "candidate_starvation_count": 0,
        },
        "feasibility_report": {
            "invalid_local_attempt_rate": 0.0,
            "sequential_feasibility_collapse_count": 0,
        },
        "metric_report": {
            "cumulative_official_like_cost_delta_by_case": {
                "19": -1e-6,
                "24": -1e-6,
                "25": -1e-6,
                "51": -1e-6,
                "76": 0.0,
                "79": 0.0,
                "91": 0.0,
                "99": 0.0,
            }
        },
        "pareto_report": {"original_inclusive_pareto_non_empty_count_by_iteration": {"0": 8}},
    }
    assert decision_for_local_iter0(reports) == "promote_to_step7c_local_lane_search"
