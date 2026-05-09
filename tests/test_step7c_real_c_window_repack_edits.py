from __future__ import annotations

import torch

from puzzleplace.alternatives.window_repack_edits import (
    STRATEGIES,
    build_window_repack_candidates,
    evaluate_window_repack_edits,
    feasibility_report,
    route_report,
    strategy_ablation_report,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase


def _case() -> FloorSetCase:
    constraints = torch.zeros((10, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    constraints[1, ConstraintColumns.MIB] = 1
    constraints[2, ConstraintColumns.MIB] = 1
    constraints[3, ConstraintColumns.CLUSTER] = 1
    constraints[4, ConstraintColumns.CLUSTER] = 1
    return FloorSetCase(
        case_id="validation-0",
        block_count=10,
        area_targets=torch.full((10,), 4.0),
        b2b_edges=torch.tensor(
            [
                [1, 9, 1.0],
                [2, 9, 1.0],
                [3, 4, 1.0],
                [5, 6, 1.0],
                [7, 8, 1.0],
            ],
            dtype=torch.float32,
        ),
        p2b_edges=torch.tensor([[0, 9, 1.0], [1, 1, 1.0]], dtype=torch.float32),
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
                [10.0, 20.0, 2.0, 2.0],
                [20.0, 20.0, 2.0, 2.0],
                [30.0, 20.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_window_repack_candidates_cover_strategies_and_preserve_anchors() -> None:
    candidates = build_window_repack_candidates([0], {0: _case()})
    assert [candidate.strategy for candidate in candidates] == list(STRATEGIES)
    for candidate in candidates:
        assert 0 not in candidate.changed_blocks
    critical = next(c for c in candidates if c.strategy == "critical_net_slack_fit_window")
    assert critical.window_blocks
    assert critical.internal_trial_count > 0


def test_window_repack_evaluation_keeps_global_report_only_and_local_route() -> None:
    rows = evaluate_window_repack_edits(build_window_repack_candidates([0], {0: _case()}))
    by_strategy = {row["strategy"]: row for row in rows}
    assert by_strategy["legacy_step7g_global_report"]["route_lane"] == "global_report_only"
    assert by_strategy["legacy_step7g_global_report"]["report_only"] is True
    assert by_strategy["critical_net_slack_fit_window"]["report_only"] is False
    assert by_strategy["critical_net_slack_fit_window"]["actual_locality_class"] != "global"
    assert by_strategy["critical_net_slack_fit_window"]["window_block_count"] >= 2


def test_window_repack_reports_required_accounting() -> None:
    rows = evaluate_window_repack_edits(build_window_repack_candidates([0], {0: _case()}))
    feasibility = feasibility_report(
        rows,
        real_case_count=1,
        real_b_feasibility={
            "official_like_cost_improving_count": 3,
            "official_like_cost_improvement_density": 0.0536,
            "official_like_hard_feasible_rate": 0.875,
        },
    )
    route = route_report(rows)
    ablation = strategy_ablation_report(rows)
    assert feasibility["real_case_count"] == 1
    assert feasibility["window_candidate_count"] == len(STRATEGIES)
    assert feasibility["average_window_block_count"] > 0.0
    assert "critical_net_slack_fit_window" in feasibility["candidate_count_by_strategy"]
    assert "official_like_cost_improvement_density" in feasibility
    assert "no_feasible_slot_count" in feasibility
    assert route["global_report_only_count"] == 1
    assert "balanced_window_swap" in ablation
