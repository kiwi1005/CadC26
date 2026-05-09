from __future__ import annotations

from puzzleplace.alternatives.locality_routing import (
    actual_locality_from_step7f,
    actual_weak_labels_from_step7f,
    calibration_report,
    predict_move_locality,
    routing_quality_report,
)
from puzzleplace.diagnostics.spatial_locality import build_locality_maps, touched_region_stats
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame
from scripts.step7g_run_spatial_locality_routing import _visualization_audit


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


def test_routing_quality_preserves_nonlocal_safe_improvement() -> None:
    predictions = [
        {
            "case_id": 19,
            "predicted_locality_class": "global",
            "predicted_repair_mode": "global_route_not_local_selector",
            "source_move_type": "shape_probe",
            "hard_invalid_before_repair": False,
        }
    ]
    rows = {
        19: [
            {
                "repair_mode": "current_repair_baseline",
                "moved_block_fraction": 0.9,
                "affected_region_count": 16,
                "MIB_group_violation_after": 0,
                "repair_radius_exceeded": True,
                "hard_feasible_after": True,
            }
        ]
    }
    pareto = {
        "19": {
            "front": [
                {
                    "repair_mode": "current_repair_baseline",
                    "hard_feasible_after": True,
                    "repair_radius_exceeded": False,
                    "hpwl_delta_norm": -0.1,
                    "bbox_delta_norm": 0.0,
                    "boundary_delta": 0.0,
                },
                {
                    "repair_mode": "rollback_to_original",
                    "hard_feasible_after": True,
                    "repair_radius_exceeded": False,
                },
            ]
        }
    }

    quality = routing_quality_report(predictions, rows, pareto)

    assert quality["invalid_local_repair_attempt_rate_before_routing"] == 1.0
    assert quality["invalid_local_repair_attempt_rate_after_routing"] == 0.0
    assert quality["safe_improvement_count_after_routing_preserved"] == 1
    assert quality["safe_improvement_count_after_routing_local_selector_only"] == 0
    assert quality["useful_improvements_requiring_nonlocal_followup"] == [19]
    assert quality["useful_improvements_lost_by_over_aggressive_prediction"] == []


def test_visualization_audit_marks_reconstructed_trace_confidence(tmp_path) -> None:
    viz_dir = tmp_path / "step7f_visualizations"
    viz_dir.mkdir()
    (viz_dir / "case099_region_cell_repair.png").write_bytes(b"png")
    (viz_dir / "case091_region_cell_repair.png").write_bytes(b"png")
    (viz_dir / "arrow_endpoint_debug.json").write_text(
        """[
          {
            "case_id": 99,
            "repair_mode": "region_cell_repair",
            "block_id": 1,
            "raw_before_center": [0.0, 0.0],
            "raw_after_center": [3.0, 4.0],
            "drawn_start": [2.4, 3.2],
            "drawn_end": [3.0, 4.0],
            "distance": 5.0,
            "after_inside_frame": false,
            "block_id_matched": true
          }
        ]"""
    )

    audit = _visualization_audit(viz_dir)

    assert audit["trace_confidence"] == "reconstructed"
    assert audit["arrow_endpoint_is_after_center"] is True
    assert audit["raw_distance_matches_centers"] is True
    assert audit["arrows_do_not_autoscale_plot_into_unreadability"] is True
    assert audit["suspicious_pngs"][0]["exists"] is True
