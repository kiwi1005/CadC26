from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns
from puzzleplace.research.boundary_failure_attribution import classify_boundary_failures
from puzzleplace.research.move_library import (
    MoveCandidate,
    build_case_suite,
    evaluate_move,
    generate_move_candidates,
    profile_case,
    select_case_alternative,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_case_suite_and_profile_emit_required_tags() -> None:
    cases = [make_step6g_synthetic_case(case_id=i, block_count=21 + i) for i in range(6)]
    suite = build_case_suite(cases, diagnostic_count=3, holdout_count=2)
    placements = {
        idx: (float(idx * 3), 0.0, 2.0, 2.0)
        for idx in range(cases[0].block_count)
    }

    profile = profile_case(cases[0], placements, suite="diagnostic")

    assert suite["diagnostic_case_ids"] == [0, 1, 2]
    assert suite["holdout_case_ids"] == [3, 4]
    assert profile["n_blocks"] == 21
    assert "baseline_boundary_failure_rate" in profile
    assert "pin_box_aspect" in profile


def test_generate_move_candidates_uses_failure_targets_and_library() -> None:
    case = make_step6g_synthetic_case(block_count=6)
    placements = {idx: (float(idx * 3), 0.0, 2.0, 2.0) for idx in range(6)}
    failures = [{"block_id": 2, "final_edge_owner_block_ids": [0]}]

    moves = generate_move_candidates(case, placements, failures, top_k_targets=2)

    assert moves[0].move_type == "simple_compaction"
    assert any(move.move_type == "boundary_edge_reassign" for move in moves)
    assert any(2 in move.target_blocks for move in moves)


def test_evaluate_move_outputs_required_fields_and_safe_rejects_no_effect() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.area_targets = torch.full((4,), 4.0)
    placements = {
        0: (0.0, 0.0, 2.0, 2.0),
        1: (2.0001, 0.0, 2.0, 2.0),
        2: (4.0002, 0.0, 2.0, 2.0),
        3: (6.0003, 0.0, 2.0, 2.0),
    }
    frame = PuzzleFrame(0.0, 0.0, 12.0, 8.0, density=0.9, variant="unit")

    row = evaluate_move(
        case,
        placements,
        frame,
        MoveCandidate("mib_master_aspect_flip", (0,), "unit-test"),
        mode="safe",
    )

    for key in (
        "case_id",
        "move_type",
        "target_blocks",
        "before_metrics",
        "after_metrics",
        "accepted",
        "rejected_reason",
        "boundary_delta",
        "bbox_delta",
        "hpwl_delta",
        "soft_delta",
        "grouping_delta",
        "mib_delta",
        "hard_feasible",
        "frame_protrusion",
        "generation_time_ms",
        "repair_time_ms",
        "eval_time_ms",
        "num_candidates_evaluated",
        "improvement_per_ms",
    ):
        assert key in row
    assert row["accepted"] is False
    assert "target_not_mib" in row["rejected_reason"]


def test_select_case_alternative_prefers_positive_boundary_move() -> None:
    rows = [
        {
            "case_id": 0,
            "move_type": "simple_compaction",
            "target_blocks": [1],
            "accepted": True,
            "selection_reason": "accepted_by_safe_gate",
            "boundary_delta": 0.1,
            "bbox_delta": -10.0,
            "hpwl_delta": 0.0,
            "soft_delta": -1.0,
            "grouping_delta": 0.0,
            "mib_delta": 0.0,
            "hard_feasible": True,
            "frame_protrusion": 0.0,
            "generation_time_ms": 1.0,
            "repair_time_ms": 0.0,
            "eval_time_ms": 1.0,
        }
    ]

    selected = select_case_alternative(rows)

    assert selected["selected_move_type"] == "simple_compaction"
    assert selected["boundary_delta"] == 0.1


def test_failure_rows_can_feed_candidate_generation() -> None:
    case = make_step6g_synthetic_case(block_count=5)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    placements = {
        0: (1.0, 0.0, 2.0, 2.0),
        1: (4.0, 0.0, 2.0, 2.0),
        2: (1.0, 3.0, 2.0, 2.0),
        3: (-3.0, 6.0, 1.0, 1.0),
        4: (6.0, 0.0, 1.0, 1.0),
    }
    hull = PuzzleFrame(0.0, 0.0, 8.0, 8.0, density=0.9, variant="hull")
    failures = classify_boundary_failures(case, placements, placements, predicted_hull=hull)

    moves = generate_move_candidates(case, placements, failures, top_k_targets=3)

    assert failures
    assert moves
