from __future__ import annotations

import torch

from puzzleplace.alternatives.objective_corridors import (
    ObjectiveVector,
    accepted_gates,
    grid_from_positions,
    objective_vector_for_candidate,
    positions_from_case_targets,
    rejection_reasons,
)
from puzzleplace.data.schema import FloorSetCase


def _synthetic_case() -> FloorSetCase:
    return FloorSetCase(
        case_id=8,
        block_count=4,
        area_targets=torch.tensor([4.0, 4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0]]),
        pins_pos=torch.tensor([[9.0, 1.0]]),
        constraints=torch.zeros((4, 5)),
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 0.0, 2.0, 2.0],
                [20.0, 0.0, 2.0, 2.0],
                [30.0, 0.0, 2.0, 2.0],
            ]
        ),
    )


def test_wire_safe_objective_vector_accepts_non_regressing_move() -> None:
    case = _synthetic_case()
    positions = positions_from_case_targets(case)
    frame = grid_from_positions(positions, grid_size=8)
    target_box = (3.0, 0.0, 2.0, 2.0)

    vector = objective_vector_for_candidate(
        case,
        positions,
        block_id=0,
        target_box=target_box,
        frame=frame,
        candidate={"source_family": "unit", "heatmap_score": 0.25},
    )

    assert vector.hpwl_delta_proxy < 0.0
    assert vector.bbox_area_delta_proxy < 0.0
    assert vector.overlap_risk_proxy == 0.0
    assert vector.heatmap_support == 0.25
    assert all(accepted_gates(vector, mode="wire_safe").values())
    assert all(accepted_gates(vector, mode="bbox_shrink_wire_safe").values())


def test_rejection_reasons_and_soft_budget_gate_are_vector_explicit() -> None:
    regressing = ObjectiveVector(
        hpwl_delta_proxy=3.0,
        bbox_area_delta_proxy=2.0,
        boundary_delta_proxy=1.0,
        group_delta_proxy=0.0,
        mib_delta_proxy=0.0,
        overlap_risk_proxy=1.0,
        displacement=4.0,
        expanded_net_count=1,
    )
    assert accepted_gates(regressing, mode="wire_safe") == {
        "overlap": False,
        "hpwl": False,
        "bbox": False,
        "soft": False,
    }
    assert set(rejection_reasons(regressing)) == {
        "overlap_risk",
        "hpwl_regression_proxy",
        "bbox_regression_proxy",
        "boundary_regression_proxy",
    }

    soft_repair = ObjectiveVector(
        hpwl_delta_proxy=0.5,
        bbox_area_delta_proxy=0.0,
        boundary_delta_proxy=-2.0,
        group_delta_proxy=0.0,
        mib_delta_proxy=0.0,
        overlap_risk_proxy=0.0,
        displacement=20.0,
        expanded_net_count=0,
    )
    assert all(accepted_gates(soft_repair, mode="soft_repair_budgeted").values())
