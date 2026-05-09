from __future__ import annotations

import torch

from puzzleplace.alternatives.pareto_slack_fit_edits import (
    build_pareto_slack_fit_candidates,
    constrained_pareto_front,
    dominance_report,
    evaluate_pareto_slack_fit_edits,
    feasibility_and_improvement_report,
    objective_vector_table,
    pareto_front_report,
    route_report,
    sensitivity_report,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase


def _case() -> FloorSetCase:
    constraints = torch.zeros((8, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    return FloorSetCase(
        case_id="validation-0",
        block_count=8,
        area_targets=torch.full((8,), 4.0),
        b2b_edges=torch.tensor(
            [[1, 7, 1.0], [2, 7, 1.0], [3, 4, 1.0], [5, 6, 1.0]],
            dtype=torch.float32,
        ),
        p2b_edges=torch.tensor([[0, 7, 1.0], [1, 1, 1.0]], dtype=torch.float32),
        pins_pos=torch.tensor([[12.0, 0.0], [8.0, 0.0]], dtype=torch.float32),
        constraints=constraints,
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 0.0, 2.0, 2.0],
                [20.0, 0.0, 2.0, 2.0],
                [0.0, 10.0, 2.0, 2.0],
                [10.0, 10.0, 2.0, 2.0],
                [20.0, 10.0, 2.0, 2.0],
                [0.0, 20.0, 2.0, 2.0],
                [20.0, 20.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_pareto_slack_fit_candidates_expand_local_lane_and_preserve_anchor() -> None:
    candidates = build_pareto_slack_fit_candidates([0], {0: _case()})
    strategies = {candidate.strategy for candidate in candidates}
    assert "original_layout" in strategies
    assert "hpwl_directed_local_nudge_expanded" in strategies
    assert "slack_fit_insertion_expanded" in strategies
    assert "nearby_slack_slot_insertion" in strategies
    assert "legacy_step7g_global_report" in strategies
    assert all(0 not in candidate.changed_blocks for candidate in candidates)
    assert len(candidates) > 10


def test_pareto_slack_fit_route_and_objective_reports() -> None:
    rows = evaluate_pareto_slack_fit_edits(build_pareto_slack_fit_candidates([0], {0: _case()}))
    route = route_report(rows)
    vectors = objective_vector_table(rows)
    assert route["global_report_only_count"] == 1
    assert route["invalid_local_attempt_rate"] == 0.0
    assert len(vectors) == len(rows)
    assert all("objective_vector" in row for row in vectors)


def test_constrained_pareto_accounting_includes_original_and_dominance() -> None:
    rows = evaluate_pareto_slack_fit_edits(build_pareto_slack_fit_candidates([0], {0: _case()}))
    front = constrained_pareto_front(rows)
    pareto = pareto_front_report(rows)
    dominance = dominance_report(rows)
    feasibility = feasibility_and_improvement_report(rows, pareto, dominance)
    sensitivity = sensitivity_report(rows)
    assert front
    assert pareto["original_inclusive_pareto_non_empty_count"] == 1
    assert "candidates_dominated_by_original_count" in dominance
    assert feasibility["real_case_count"] == 1
    assert "hpwl_gain_to_official_like_conversion_rate" in feasibility
    assert sensitivity["sensitivity_to_epsilon_or_tolerance"]
