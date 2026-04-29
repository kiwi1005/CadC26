from __future__ import annotations

from puzzleplace.alternatives.locality_routing import (
    actual_locality_from_step7f,
    actual_weak_labels_from_step7f,
    calibration_report,
    predict_move_locality,
)
from puzzleplace.diagnostics.spatial_locality import build_locality_maps, touched_region_stats
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_locality_maps_emit_coarse_and_adaptive_channels() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=16)
    frame = PuzzleFrame(0.0, 0.0, 160.0, 100.0, density=0.8, variant="unit")
    placements = {
        idx: (float(idx % 4) * 20.0, float(idx // 4) * 16.0, 8.0, 8.0)
        for idx in range(16)
    }

    maps = build_locality_maps(case, placements, frame)

    assert [row["name"] for row in maps["resolutions"]] == ["coarse", "adaptive"]
    assert "repair_reachability_mask" in maps["resolutions"][0]["regions"][0]
    assert "max_occupancy_delta_adaptive_minus_coarse" in maps["sensitivity"]


def test_touched_region_stats_reports_macro_closure() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=16)
    frame = PuzzleFrame(0.0, 0.0, 160.0, 100.0, density=0.8, variant="unit")
    placements = {
        idx: (float(idx % 4) * 20.0, float(idx // 4) * 16.0, 8.0, 8.0)
        for idx in range(16)
    }

    stats = touched_region_stats(case, placements, frame, {0, 1, 2})

    assert stats["touched_region_count"] >= 1
    assert stats["macro_closure_size"] >= 3


def test_predict_move_locality_routes_global_move() -> None:
    prediction = predict_move_locality(
        case_id=1,
        block_count=100,
        changed_block_count=70,
        touched_region_count=12,
        macro_closure_size=72,
        min_region_slack=0.0,
        free_space_fit_ratio=3.0,
        hard_summary={"hard_feasible": False},
    )

    assert prediction["predicted_locality_class"] == "global"
    assert prediction["predicted_repair_mode"] == "global_route_not_local_selector"


def test_calibration_counts_correct_global() -> None:
    predictions = [
        {
            "case_id": 1,
            "predicted_locality_class": "global",
            "predicted_repair_mode": "global_route_not_local_selector",
        }
    ]
    rows = {
        1: [
            {
                "repair_mode": "current_repair_baseline",
                "moved_block_fraction": 1.0,
                "affected_region_count": 16,
                "MIB_group_violation_after": 0,
                "repair_radius_exceeded": False,
            }
        ]
    }

    report = calibration_report(predictions, rows)

    assert actual_locality_from_step7f(rows[1]) == "global"
    assert actual_weak_labels_from_step7f(rows[1])["actual_radius_exceeded"] is False
    assert report["counts"]["correct_global"] == 1
    assert report["rows"][0]["actual_moved_block_fraction"] == 1.0
