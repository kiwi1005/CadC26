from __future__ import annotations

import torch

from puzzleplace.alternatives.metric_directed_edits import (
    STRATEGIES,
    build_metric_directed_edit_candidates,
    evaluate_metric_directed_edits,
    feasibility_report,
    route_report,
    strategy_ablation_report,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase


def _case() -> FloorSetCase:
    constraints = torch.zeros((8, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    constraints[1, ConstraintColumns.MIB] = 1
    constraints[2, ConstraintColumns.MIB] = 1
    constraints[3, ConstraintColumns.CLUSTER] = 1
    constraints[4, ConstraintColumns.CLUSTER] = 1
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


def test_metric_directed_candidates_cover_strategies_and_preserve_anchors() -> None:
    candidates = build_metric_directed_edit_candidates([0], {0: _case()})
    assert [candidate.strategy for candidate in candidates] == list(STRATEGIES)
    for candidate in candidates:
        assert 0 not in candidate.changed_blocks


def test_metric_directed_evaluation_keeps_global_report_only() -> None:
    rows = evaluate_metric_directed_edits(build_metric_directed_edit_candidates([0], {0: _case()}))
    by_strategy = {row["strategy"]: row for row in rows}
    assert by_strategy["legacy_step7g_global_report"]["route_lane"] == "global_report_only"
    assert by_strategy["legacy_step7g_global_report"]["report_only"] is True
    assert by_strategy["hpwl_directed_local_nudge"]["report_only"] is False
    assert by_strategy["hpwl_directed_local_nudge"]["actual_locality_class"] != "global"


def test_metric_directed_reports_required_accounting() -> None:
    rows = evaluate_metric_directed_edits(build_metric_directed_edit_candidates([0], {0: _case()}))
    feasibility = feasibility_report(
        rows,
        real_case_count=1,
        real_a_feasibility={
            "actual_safe_improvement_count": 0,
            "official_like_hard_feasible_rate": 1.0,
            "real_non_global_candidate_rate": 1.0,
        },
    )
    route = route_report(rows)
    ablation = strategy_ablation_report(rows)
    assert feasibility["real_case_count"] == 1
    assert "hpwl_directed_local_nudge" in feasibility["candidate_count_by_strategy"]
    assert "official_like_cost_improving_count" in feasibility
    assert route["global_report_only_count"] == 1
    assert "slack_fit_insertion" in ablation
