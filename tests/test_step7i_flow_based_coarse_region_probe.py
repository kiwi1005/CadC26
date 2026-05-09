from __future__ import annotations

import torch

from puzzleplace.alternatives.coarse_region_flow import (
    assignment_cost_components,
    build_flow_assignment_candidates,
    build_region_maps,
    coarse_assignment_rows,
    decision_for_step7i,
    evaluate_flow_assignment_candidates,
    feasibility_report,
)
from puzzleplace.alternatives.real_placement_edits import frame_from_baseline, placements_from_case
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.region_topology import build_region_grid, region_occupancy


def _case() -> FloorSetCase:
    constraints = torch.zeros((10, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    constraints[3, ConstraintColumns.CLUSTER] = 1
    constraints[4, ConstraintColumns.CLUSTER] = 1
    return FloorSetCase(
        case_id=0,
        block_count=10,
        area_targets=torch.full((10,), 4.0),
        b2b_edges=torch.tensor(
            [[1, 2, 2.0], [1, 3, 1.5], [3, 4, 2.0], [5, 6, 1.0], [7, 8, 1.0]],
            dtype=torch.float32,
        ),
        p2b_edges=torch.tensor([[0, 1, 1.0], [1, 3, 1.0], [2, 8, 1.0]], dtype=torch.float32),
        pins_pos=torch.tensor([[30.0, 0.0], [0.0, 30.0], [30.0, 30.0]], dtype=torch.float32),
        constraints=constraints,
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [6.0, 0.0, 2.0, 2.0],
                [12.0, 0.0, 2.0, 2.0],
                [0.0, 6.0, 2.0, 2.0],
                [6.0, 6.0, 2.0, 2.0],
                [12.0, 6.0, 2.0, 2.0],
                [0.0, 12.0, 2.0, 2.0],
                [6.0, 12.0, 2.0, 2.0],
                [12.0, 12.0, 2.0, 2.0],
                [18.0, 12.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_region_map_and_assignment_rows_are_deterministic() -> None:
    case = _case()
    maps = build_region_maps([0], {0: case})
    assert maps["coarse_region_count_by_case"] == {"0": 16}
    placements = placements_from_case(case)
    rows = coarse_assignment_rows(case, placements, frame_from_baseline(placements))
    assert rows
    assert all("assignment_cost" in row for row in rows)
    assert rows == coarse_assignment_rows(case, placements, frame_from_baseline(placements))


def test_assignment_cost_reports_required_components() -> None:
    case = _case()
    placements = placements_from_case(case)
    frame = frame_from_baseline(placements)
    cell = build_region_grid(frame, rows=4, cols=4)[0]
    region = region_occupancy(case, placements, frame, rows=4, cols=4)["regions"][0]
    costs = assignment_cost_components(
        case,
        placements,
        frame,
        [1, 2],
        source_region="r0_0",
        target_region=str(cell["region_id"]),
        target_cell=cell,
        region=region,
        current_center=(1.0, 1.0),
        demand_center=(2.0, 2.0),
    )
    assert set(costs) == {
        "hpwl_community_demand",
        "region_slack_capacity",
        "bbox_pressure",
        "fixed_preplaced_incompatibility",
        "mib_group_closure_risk",
        "boundary_ownership_compatibility",
    }


def test_flow_candidates_evaluate_routes_and_feasibility_summary() -> None:
    case = _case()
    rows = evaluate_flow_assignment_candidates(build_flow_assignment_candidates([0], {0: case}))
    report = feasibility_report(rows)
    assert report["assignment_candidate_count"] == len(rows)
    assert report["regional_candidate_count"] + report["global_candidate_count"] >= 1
    assert "candidate_count_by_assignment_type" in report


def test_decision_builds_regional_legalizer_for_promising_but_infeasible_assignments() -> None:
    feasibility = {
        "official_like_improving_candidate_count": 0,
        "regional_candidate_count": 4,
        "regional_feasible_or_legalizer_promising_count": 4,
        "official_like_hard_feasible_rate": 0.2,
        "non_global_candidate_rate": 0.8,
    }
    starvation = {"local_starvation_case_recovery_count": 0}
    corr = {"assignment_cost_component_correlation_with_actual_delta": {"x": {"pearson": 0.2}}}
    assert decision_for_step7i(feasibility, starvation, corr) == "build_regional_legalizer"
