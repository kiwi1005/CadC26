from __future__ import annotations

import math

from puzzleplace.actions import ExecutionState
from puzzleplace.research.puzzle_candidate_payload import build_puzzle_candidate_descriptors
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import (
    PuzzleFrame,
    estimate_virtual_puzzle_frame,
    frame_diagnostics,
    multistart_virtual_frames,
)


def test_virtual_frame_contains_pins_preplaced_and_area_budget() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    frame = estimate_virtual_puzzle_frame(case, target_density=0.90)

    for px, py in case.pins_pos.tolist():
        assert frame.xmin <= float(px) <= frame.xmax
        assert frame.ymin <= float(py) <= frame.ymax
    assert case.target_positions is not None
    preplaced = tuple(float(v) for v in case.target_positions[1].tolist())
    assert frame.contains_box(preplaced)
    assert frame.area >= float(case.area_targets.sum().item()) / 0.90 - 1e-5

    variants = [row.variant for row in multistart_virtual_frames(case)]
    assert {"tight", "medium", "loose", "pin_aspect", "square"} <= set(variants)


def test_frame_aware_candidates_keep_full_rectangle_inside_frame_and_snap_boundary() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    frame = estimate_virtual_puzzle_frame(case)
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[2],
        max_shape_bins=5,
        max_descriptors_per_block=32,
        virtual_frame=frame,
    )

    assert descriptors
    assert all(
        frame.contains_box(
            (
                float(desc.action_token.x),
                float(desc.action_token.y),
                float(desc.action_token.w),
                float(desc.action_token.h),
            )
        )
        for desc in descriptors
    )
    boundary_candidates = [
        desc
        for desc in descriptors
        if desc.action_token.metadata.get("site_key") == "frame_boundary_left"
    ]
    assert boundary_candidates
    assert all(float(desc.action_token.x) == frame.xmin for desc in boundary_candidates)


def test_frame_default_shape_bins_are_mild_before_extreme_fallback() -> None:
    case = make_step6g_synthetic_case(block_count=8)
    frame = estimate_virtual_puzzle_frame(case)
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[3],
        max_shape_bins=None,
        virtual_frame=frame,
    )

    assert descriptors
    log_aspects = [
        math.log(float(desc.action_token.w) / float(desc.action_token.h))
        for desc in descriptors
        if not desc.exact_shape_flag
    ]
    assert log_aspects
    assert max(abs(value) for value in log_aspects) <= 2.0 + 1e-5


def test_empty_frame_candidate_pool_relaxes_before_returning_empty() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    frame = PuzzleFrame(0.0, 0.0, 1.0, 1.0, density=0.90, variant="tiny")
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[0],
        max_shape_bins=None,
        max_descriptors_per_block=8,
        virtual_frame=frame,
        frame_relaxation_steps=1,
        frame_expand_factor=2.0,
    )

    assert descriptors
    assert {desc.action_token.metadata["frame_relaxation"] for desc in descriptors} == {1}


def test_frame_diagnostics_report_protrusion_metrics() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    frame = PuzzleFrame(0.0, 0.0, 4.0, 4.0, density=0.90, variant="unit-test")
    metrics = frame_diagnostics(
        case,
        {
            0: (0.0, 0.0, 2.0, 2.0),
            1: (3.5, 1.0, 2.0, 2.0),
        },
        frame,
    )

    assert metrics["num_frame_violations"] == 1
    assert metrics["outside_frame_area_ratio"] > 0.0
    assert metrics["max_protrusion_distance"] == 1.5


def test_step6i_boundary_commitment_keeps_only_frame_satisfying_candidates() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    frame = estimate_virtual_puzzle_frame(case)
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[2],
        max_shape_bins=5,
        max_descriptors_per_block=64,
        virtual_frame=frame,
        commit_boundary_to_frame=True,
    )

    assert descriptors
    assert all(
        float(desc.action_token.metadata["boundary_frame_satisfaction"]) == 1.0
        for desc in descriptors
    )
    assert all(float(desc.action_token.x) == frame.xmin for desc in descriptors)


def test_step6i_boundary_metrics_count_frame_committed_edges() -> None:
    case = make_step6g_synthetic_case(block_count=10)
    frame = PuzzleFrame(0.0, 0.0, 10.0, 10.0, density=0.90, variant="unit-test")
    metrics = frame_diagnostics(
        case,
        {
            0: (1.0, 1.0, 2.0, 2.0),
            1: (3.0, 1.0, 2.0, 2.0),
            2: (0.0, 3.0, 2.0, 2.0),  # synthetic block 2 has left boundary
            9: (3.0, 3.0, 2.0, 2.0),  # synthetic block 9 has boundary but misses left
        },
        frame,
    )

    assert metrics["boundary_frame_total_edges"] == 2
    assert metrics["boundary_frame_satisfied_edges"] == 1
    assert metrics["boundary_frame_satisfaction_rate"] == 0.5
    assert metrics["boundary_frame_unsatisfied_blocks"] == 1


def test_step6j_predicted_hull_candidates_are_kept_without_hard_frame_commit() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    frame = estimate_virtual_puzzle_frame(case)
    # Smaller than the frame so Step6J can distinguish hull ownership from frame containment.
    hull = PuzzleFrame(
        frame.xmin + 1.0,
        frame.ymin + 1.0,
        frame.xmax - 1.0,
        frame.ymax - 1.0,
        density=0.90,
        variant="unit-hull",
    )
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[2],
        max_shape_bins=3,
        max_descriptors_per_block=64,
        virtual_frame=frame,
        predicted_hull=hull,
        boundary_commit_mode="prefer_predicted_hull",
    )

    assert descriptors
    predicted_values = [
        float(desc.action_token.metadata["predicted_hull_satisfaction"]) for desc in descriptors
    ]
    frame_values = [
        float(desc.action_token.metadata["boundary_frame_satisfaction"]) for desc in descriptors
    ]
    assert any(value == 1.0 for value in predicted_values)
    assert any(value == 1.0 for value in frame_values)
    assert any(value < 1.0 for value in predicted_values)
