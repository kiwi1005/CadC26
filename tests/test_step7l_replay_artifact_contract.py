from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.experiments.step7l_learning_guided_replay import (
    nearest_nonoverlap_position,
    obstacle_aware_single_block_attempt,
    summarize_replay_rows,
)


def _case() -> FloorSetCase:
    return FloorSetCase(
        case_id=24,
        block_count=3,
        area_targets=torch.tensor([4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        p2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        pins_pos=torch.tensor([[-1.0, -1.0]]),
        constraints=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [4.0, 0.0, 2.0, 2.0],
                [8.0, 0.0, 2.0, 2.0],
            ]
        ),
        metrics=torch.zeros(8),
    )


def test_nearest_nonoverlap_position_avoids_existing_boxes() -> None:
    positions = [(0.0, 0.0, 2.0, 2.0), (4.0, 0.0, 2.0, 2.0), (8.0, 0.0, 2.0, 2.0)]
    window = {"cx": 5.0, "cy": 1.0, "w": 2.0, "h": 2.0, "frame": {"x": 0, "y": 0, "w": 12, "h": 8}}
    candidate = nearest_nonoverlap_position(positions, 0, target_center=(5.0, 1.0), window=window)
    assert candidate is not None
    assert candidate != positions[0]
    assert candidate != positions[1]


def test_obstacle_aware_attempt_realizes_single_block_request() -> None:
    request = {
        "request_id": "req",
        "block_id": 0,
        "target_window": {
            "cx": 6.0,
            "cy": 6.0,
            "w": 2.0,
            "h": 2.0,
            "frame": {"x": 0.0, "y": 0.0, "w": 12.0, "h": 12.0},
        },
    }
    positions = [(0.0, 0.0, 2.0, 2.0), (4.0, 0.0, 2.0, 2.0), (8.0, 0.0, 2.0, 2.0)]
    attempt = obstacle_aware_single_block_attempt(request, _case(), positions)
    assert attempt.status == "realized_single_block_target_window"
    assert attempt.positions is not None
    assert attempt.moved_block_ids == [0]


def test_summarize_replay_rows_reports_phase_gates_closed_without_broad_winners() -> None:
    rows = [
        {
            "is_anchor": True,
            "case_id": "24",
            "quality_gate_status": "original_anchor",
            "failure_attribution": "original_anchor",
            "request_global_report_only": True,
        },
        {
            "is_anchor": False,
            "case_id": "24",
            "fresh_metric_available": True,
            "hard_feasible_nonnoop": True,
            "official_like_cost_improving": False,
            "quality_gate_status": "dominated_by_original",
            "metric_regressing": True,
            "hpwl_delta": 1.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "failure_attribution": "hpwl_regression",
            "actual_locality_class": "local",
            "replayed_signature": "sig1",
        },
    ]
    summary = summarize_replay_rows(rows, request_count=2)
    assert summary["decision"] == "complete_step7l_deterministic_prior_and_defer_gnn_rl"
    assert summary["phase3_gnn_gate_open"] is False
    assert summary["phase4_offline_rl_gate_open"] is False
    assert summary["fresh_hard_feasible_nonnoop_count"] == 1
