from __future__ import annotations

import torch

from puzzleplace.alternatives.real_placement_edits import (
    frame_from_baseline,
    placements_from_case,
)
from puzzleplace.alternatives.regional_legalizer import (
    decision_for_step7i_r,
    evaluate_regional_legalizer_candidates,
    feasibility_report,
    legalize_step7i_candidate_row,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase


def _case() -> FloorSetCase:
    constraints = torch.zeros((6, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    constraints[2, ConstraintColumns.CLUSTER] = 1
    constraints[3, ConstraintColumns.CLUSTER] = 1
    return FloorSetCase(
        case_id=0,
        block_count=6,
        area_targets=torch.full((6,), 4.0),
        b2b_edges=torch.tensor([[1, 2, 1.0], [2, 3, 1.0], [4, 5, 1.0]], dtype=torch.float32),
        p2b_edges=torch.tensor([[0, 1, 1.0]], dtype=torch.float32),
        pins_pos=torch.tensor([[12.0, 0.0]], dtype=torch.float32),
        constraints=constraints,
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [5.0, 0.0, 2.0, 2.0],
                [10.0, 0.0, 2.0, 2.0],
                [0.0, 5.0, 2.0, 2.0],
                [5.0, 5.0, 2.0, 2.0],
                [10.0, 5.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": 0,
        "candidate_id": "case000:regional:000",
        "assignment_type": "block_flow_assignment",
        "actual_locality_class": "regional",
        "descriptor_locality_class": "regional",
        "descriptor_repair_mode": "region_repair_or_planner",
        "real_block_ids_changed": [1],
        "macro_closure_size": 1,
        "source_regions": ["r0_0"],
        "target_region": "r0_1",
        "assignment_cost": {},
        "failure_attribution": "infeasible_after_regional_edit",
        "after_route_overlap_violation_count": 1,
    }
    row.update(overrides)
    return row


def test_slot_legalizer_attempt_is_reported_and_evaluated() -> None:
    case = _case()
    baseline = placements_from_case(case)
    candidate, attempt = legalize_step7i_candidate_row(
        _row(), case, baseline, frame_from_baseline(baseline)
    )  # type: ignore[arg-type]
    rows = evaluate_regional_legalizer_candidates([candidate])
    assert attempt["legalizer_strategy"] == "slot_assignment_matching"
    assert rows[0]["source_actual_locality_class"] == "regional"
    assert "after_route_official_like_hard_feasible" in rows[0]


def test_macro_rows_are_reported_as_supernode_required() -> None:
    case = _case()
    baseline = placements_from_case(case)
    candidate, attempt = legalize_step7i_candidate_row(
        _row(actual_locality_class="macro", macro_closure_size=4),  # type: ignore[arg-type]
        case,
        baseline,
        frame_from_baseline(baseline),
    )
    rows = evaluate_regional_legalizer_candidates([candidate])
    assert attempt["legalizer_status"] == "macro_closure_requires_supernode"
    assert rows[0]["failure_attribution"] == "macro_closure_requires_supernode"


def test_feasibility_report_compares_against_step7i_infeasible_baseline() -> None:
    case = _case()
    baseline = placements_from_case(case)
    candidate, _attempt = legalize_step7i_candidate_row(
        _row(), case, baseline, frame_from_baseline(baseline)
    )  # type: ignore[arg-type]
    rows = evaluate_regional_legalizer_candidates([candidate])
    report = feasibility_report(rows, [_row()])  # type: ignore[list-item]
    assert report["source_step7i_candidate_count"] == 1
    assert report["step7i_infeasible_after_regional_edit_baseline"] == 1
    assert "hard_feasible_rate_after_legalizer" in report


def test_decision_pivots_when_macro_closure_dominates() -> None:
    feasibility = {
        "hard_feasible_count_after_legalizer": 56,
        "hard_feasible_rate_after_legalizer": 1.0,
        "official_like_improving_candidate_count": 0,
        "no_op_after_legalizer_count": 48,
        "attempted_regional_candidate_count": 20,
        "attempted_macro_candidate_count": 12,
        "macro_closure_requires_supernode_count": 30,
    }
    starvation = {"local_starvation_case_recovery_count": 0}
    assert decision_for_step7i_r(feasibility, starvation) == "pivot_to_macro_closure_generator"
