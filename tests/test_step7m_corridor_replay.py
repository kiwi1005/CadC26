from __future__ import annotations

from typing import Any

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.experiments.step7m_objective_corridor_replay import (
    proxy_actual_signs,
    replay_corridor_request_row,
    summarize_corridor_replay_rows,
    target_box_from_request,
)
from puzzleplace.geometry.legality import positions_from_case_targets


def _case() -> FloorSetCase:
    return FloorSetCase(
        case_id=42,
        block_count=3,
        area_targets=torch.tensor([4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        p2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        pins_pos=torch.tensor([[-1.0, -1.0]]),
        constraints=torch.zeros((3, 5)),
        target_positions=torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 2.0, 2.0], [8.0, 0.0, 2.0, 2.0]]
        ),
        metrics=torch.zeros(8),
    )


def _fake_eval(
    _case: FloorSetCase, positions: list[tuple[float, float, float, float]]
) -> dict[str, Any]:
    x0 = min(x for x, _y, _w, _h in positions)
    x1 = max(x + w for x, _y, w, _h in positions)
    y0 = min(y for _x, y, _w, _h in positions)
    y1 = max(y + h for _x, y, _w, h in positions)
    bbox = (x1 - x0) * (y1 - y0)
    return {
        "quality": {
            "cost": bbox,
            "HPWLgap": bbox,
            "Areagap_bbox": bbox,
            "Violationsrelative": 0.0,
            "feasible": True,
        }
    }


def _request() -> dict[str, Any]:
    return {
        "request_id": "step7m-unit",
        "case_id": "42",
        "loader_index": 42,
        "gate_mode": "wire_safe",
        "source_family": "micro_axis_corridor",
        "block_id": 2,
        "move_family": "single_block_wire_safe",
        "route_class": "unrouted_objective_corridor_sidecar",
        "accepted_gates": {"overlap": True, "hpwl": True, "bbox": True, "soft": True},
        "proxy_objective_vector": {
            "hpwl_delta_proxy": -1.0,
            "bbox_area_delta_proxy": -1.0,
            "boundary_delta_proxy": 0.0,
            "group_delta_proxy": 0.0,
            "mib_delta_proxy": 0.0,
            "overlap_risk_proxy": 0.0,
        },
        "target_window": {"x": 6.0, "y": 0.0, "w": 2.0, "h": 2.0},
    }


def test_target_box_from_request_uses_exact_window_shape() -> None:
    positions = positions_from_case_targets(_case())
    assert target_box_from_request(_request(), positions) == (6.0, 0.0, 2.0, 2.0)


def test_replay_corridor_request_row_records_actual_delta_and_proxy_signs() -> None:
    case = _case()
    positions = positions_from_case_targets(case)
    before = _fake_eval(case, positions)
    row = replay_corridor_request_row(_request(), case, positions, before, _fake_eval)

    assert row["schema"] == "step7m_phase2_replay_row_v1"
    assert row["generation_status"] == "realized_exact_target_window"
    assert row["quality_gate_status"] == "archive_candidate"
    assert row["hard_feasible_nonnoop"] is True
    assert row["official_like_cost_delta"] < 0.0
    assert row["proxy_actual_signs"]["hpwl_nonregression_match"] is True
    assert row["failure_attribution"] == "none"


def test_summarize_corridor_replay_rows_keeps_gnn_rl_closed_on_tiny_corpus() -> None:
    case = _case()
    positions = positions_from_case_targets(case)
    before = _fake_eval(case, positions)
    rows = [replay_corridor_request_row(_request(), case, positions, before, _fake_eval)]
    summary = summarize_corridor_replay_rows(rows, request_count=1)

    assert summary["fresh_hard_feasible_nonnoop_count"] == 1
    assert summary["fresh_quality_gate_pass_count"] == 1
    assert summary["phase3_ablation_gate_open"] is True
    assert summary["gnn_rl_gate_open"] is False
    assert summary["proxy_actual_sign_precision"]["all_component_precision"] == 1.0


def test_proxy_actual_signs_detects_component_mismatch() -> None:
    signs = proxy_actual_signs(
        {"hpwl_delta_proxy": -1.0, "bbox_area_delta_proxy": 0.0, "boundary_delta_proxy": 0.0},
        {"hpwl_delta": 2.0, "bbox_area_delta": 0.0, "soft_constraint_delta": 0.0},
    )
    assert signs["hpwl_nonregression_match"] is False
    assert signs["bbox_nonregression_match"] is True
