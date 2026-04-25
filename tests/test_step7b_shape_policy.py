from __future__ import annotations

from puzzleplace.alternatives.shape_policy import (
    SHAPE_POLICIES,
    cap_for_block,
    capped_shape,
    pareto_front,
    posthoc_shape_probe,
    shape_policy_eval_row,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_shape_policies_include_required_alternatives() -> None:
    assert "original_shape_policy" in SHAPE_POLICIES
    assert "role_aware_cap" in SHAPE_POLICIES
    assert "boundary_edge_slot_exception" in SHAPE_POLICIES


def test_role_aware_cap_is_stricter_for_boundary_than_global() -> None:
    case = make_step6g_synthetic_case(block_count=10)

    boundary_cap = cap_for_block("role_aware_cap", case, 2)
    global_cap = cap_for_block("mild_global_cap", case, 2)

    assert boundary_cap == 1.2
    assert global_cap == 2.0


def test_capped_shape_preserves_area_and_caps_log_aspect() -> None:
    width, height = capped_shape(10.0, 10.0, 1.0, 1.5)

    assert abs(width * height - 10.0) < 1e-6
    assert abs(width / height) < 10.0


def test_posthoc_probe_emits_role_reasons_for_changed_blocks() -> None:
    case = make_step6g_synthetic_case(block_count=8)
    placements = {idx: (float(idx) * 3.0, 0.0, 8.0, 1.0) for idx in range(8)}
    frame = PuzzleFrame(0.0, 0.0, 80.0, 20.0, density=0.8, variant="unit")

    alternative, reasons = posthoc_shape_probe("boundary_strict_cap", case, placements, frame)

    assert alternative.keys() == placements.keys()
    assert reasons
    assert all("role_reason" in row for row in reasons)


def test_pareto_front_keeps_original_when_bad_policy_dominated() -> None:
    base = {
        "case_id": 0,
        "track": "posthoc_shape_probe",
        "policy": "original_shape_policy",
        "boundary_violation_delta": 0.0,
        "hpwl_delta_norm": 0.0,
        "bbox_delta_norm": 0.0,
        "aspect_pathology_score": 0.0,
        "hole_fragmentation": 0.0,
        "disruption": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
    }
    bad = {**base, "policy": "mild_global_cap", "hpwl_delta_norm": 1.0, "disruption": 1.0}

    front = pareto_front([base, bad])

    assert [row["policy"] for row in front] == ["original_shape_policy"]


def test_shape_policy_eval_row_outputs_objectives() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    baseline = {idx: (float(idx) * 3.0, 0.0, 2.0, 2.0) for idx in range(4)}
    frame = PuzzleFrame(0.0, 0.0, 20.0, 10.0, density=0.8, variant="unit")

    row = shape_policy_eval_row(
        case=case,
        policy="original_shape_policy",
        track="posthoc_shape_probe",
        baseline=baseline,
        alternative=baseline,
        frame=frame,
        role_cap_reasons=[],
    )

    assert row["policy"] == "original_shape_policy"
    assert "aspect_pathology_score" in row
