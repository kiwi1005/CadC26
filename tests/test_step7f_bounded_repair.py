from __future__ import annotations

from puzzleplace.diagnostics.repair_radius import (
    changed_blocks,
    hard_summary,
    pareto_repair_selection,
)
from puzzleplace.legalization import bounded_repair
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_bounded_repair_rollback_preserves_baseline() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=12)
    frame = PuzzleFrame(0.0, 0.0, 120.0, 80.0, density=0.8, variant="unit")
    baseline = {
        idx: (float(idx % 4) * 20.0, float(idx // 4) * 16.0, 10.0, 10.0)
        for idx in range(12)
    }
    candidate = {idx: (box[0] + 5.0, box[1], box[2], box[3]) for idx, box in baseline.items()}

    result = bounded_repair(
        case,
        baseline=baseline,
        candidate=candidate,
        frame=frame,
        mode="rollback_to_original",
    )

    assert result.placements == baseline
    assert changed_blocks(baseline, result.placements) == set()


def test_cascade_cap_marks_radius_exceeded_for_global_candidate() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=20)
    frame = PuzzleFrame(0.0, 0.0, 200.0, 100.0, density=0.8, variant="unit")
    baseline = {idx: (float(idx) * 12.0, 0.0, 8.0, 8.0) for idx in range(20)}
    candidate = {idx: (0.0, 0.0, 8.0, 8.0) for idx in range(20)}

    result = bounded_repair(
        case,
        baseline=baseline,
        candidate=candidate,
        frame=frame,
        mode="cascade_capped_repair",
        max_moved_fraction=0.20,
    )

    assert result.repair_radius_exceeded is True
    assert result.reject_reason == "repair_radius_exceeded"
    assert result.placements == baseline


def test_pareto_repair_selection_prefers_feasible_low_radius() -> None:
    rows = [
        {
            "case_id": 1,
            "source_move_type": "x",
            "repair_mode": "rollback_to_original",
            "hard_feasible_after": True,
            "repair_radius_exceeded": False,
            "frame_protrusion_after": 0.0,
            "moved_block_fraction": 0.0,
            "affected_region_count": 0,
            "hpwl_delta_norm": 0.0,
            "bbox_delta_norm": 0.0,
            "boundary_delta": 0.0,
            "reject_reason": None,
        },
        {
            "case_id": 1,
            "source_move_type": "x",
            "repair_mode": "graph_hop_repair",
            "hard_feasible_after": False,
            "repair_radius_exceeded": False,
            "frame_protrusion_after": 0.0,
            "moved_block_fraction": 0.2,
            "affected_region_count": 2,
            "hpwl_delta_norm": -0.1,
            "bbox_delta_norm": 0.0,
            "boundary_delta": 0.0,
            "reject_reason": None,
        },
    ]

    selection = pareto_repair_selection(rows)

    assert selection["representatives"]["min_radius"]["repair_mode"] == "rollback_to_original"


def test_hard_summary_reports_overlap() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=4)
    placements = {idx: (0.0, 0.0, 8.0, 8.0) for idx in range(4)}

    summary = hard_summary(case, placements)

    assert summary["overlap_count"] > 0
