from __future__ import annotations

import torch

from puzzleplace.alternatives.real_placement_edits import (
    build_real_edit_candidates,
    confusion_report,
    evaluate_real_edits,
    feasibility_report,
    metric_report,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase


def _case() -> FloorSetCase:
    constraints = torch.zeros((6, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.PREPLACED] = 1
    constraints[1, ConstraintColumns.MIB] = 1
    constraints[2, ConstraintColumns.MIB] = 1
    constraints[3, ConstraintColumns.CLUSTER] = 1
    constraints[4, ConstraintColumns.CLUSTER] = 1
    return FloorSetCase(
        case_id="validation-0",
        block_count=6,
        area_targets=torch.full((6,), 4.0),
        b2b_edges=torch.tensor(
            [[1, 5, 1.0], [2, 5, 1.0], [3, 4, 1.0], [4, 5, 1.0]],
            dtype=torch.float32,
        ),
        p2b_edges=torch.tensor([[0, 5, 1.0], [1, 1, 1.0]], dtype=torch.float32),
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
            ],
            dtype=torch.float32,
        ),
    )


def _descriptor(
    family: str, locality: str, changed: int, regions: int, macro: int
) -> dict[str, object]:
    route = {
        "local": "bounded_repair_pareto",
        "regional": "region_repair_or_planner",
        "macro": "macro_legalizer",
        "global": "global_route_not_local_selector",
    }[locality]
    return {
        "case_id": 0,
        "candidate_id": f"case000:{family}",
        "family": family,
        "predicted_locality_class": locality,
        "predicted_repair_mode": route,
        "predicted_affected_blocks": changed,
        "predicted_affected_regions": regions,
        "predicted_macro_closure_size": macro,
    }


def test_real_edit_candidates_preserve_fixed_preplaced_blocks() -> None:
    descriptors = [
        _descriptor("original_layout", "local", 0, 0, 0),
        _descriptor("vacancy_aware_local_insertion", "local", 1, 1, 1),
        _descriptor("legacy_step7g_global_move", "global", 5, 8, 5),
    ]
    candidates = build_real_edit_candidates(descriptors, {0: _case()})
    assert len(candidates) == 3
    for candidate in candidates:
        assert 0 not in candidate.changed_blocks


def test_real_edit_evaluation_keeps_global_report_only_and_local_safe() -> None:
    descriptors = [
        _descriptor("original_layout", "local", 0, 0, 0),
        _descriptor("vacancy_aware_local_insertion", "local", 1, 1, 1),
        _descriptor("adjacent_region_reassignment", "regional", 3, 3, 3),
        _descriptor("mib_group_closure_macro", "macro", 2, 1, 4),
        _descriptor("legacy_step7g_global_move", "global", 5, 8, 5),
    ]
    rows = evaluate_real_edits(build_real_edit_candidates(descriptors, {0: _case()}))
    by_family = {row["family"]: row for row in rows}
    assert by_family["legacy_step7g_global_move"]["route_lane"] == "global_report_only"
    assert by_family["legacy_step7g_global_move"]["report_only"] is True
    assert by_family["vacancy_aware_local_insertion"]["actual_locality_class"] != "global"
    assert by_family["vacancy_aware_local_insertion"]["report_only"] is False
    assert (
        by_family["vacancy_aware_local_insertion"]["after_route_fixed_or_preplaced_violation_count"]
        == 0
    )


def test_real_a_reports_include_required_accounting() -> None:
    descriptors = [
        _descriptor("original_layout", "local", 0, 0, 0),
        _descriptor("vacancy_aware_local_insertion", "local", 1, 1, 1),
        _descriptor("legacy_step7g_global_move", "global", 5, 8, 5),
    ]
    rows = evaluate_real_edits(build_real_edit_candidates(descriptors, {0: _case()}))
    feasibility = feasibility_report(
        rows, descriptor_candidate_count=len(descriptors), real_case_count=1
    )
    confusion = confusion_report(rows)
    metrics = metric_report(rows)
    assert feasibility["real_case_count"] == 1
    assert feasibility["real_edit_candidate_count"] == 3
    assert feasibility["invalid_local_attempt_rate"] == 0.0
    assert "official_like_cost_delta_distribution" in metrics
    assert confusion["descriptor_class_vs_actual_route_confusion"]["global"]["global"] == 1
