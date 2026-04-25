from __future__ import annotations

import torch

from puzzleplace.research.pathology import (
    METRIC_SEMANTICS,
    guard_calibration_candidates,
    label_case_pathology,
    layout_pathology_metrics,
    normalized_move_row,
    scale_coverage_report,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_metric_semantics_make_delta_direction_explicit() -> None:
    assert METRIC_SEMANTICS["hpwl_delta"]["positive_means"] == "worse"
    assert METRIC_SEMANTICS["bbox_delta"]["positive_means"] == "worse"
    assert METRIC_SEMANTICS["boundary_delta"]["positive_means"] == "better"
    assert "soft_delta < 0" in METRIC_SEMANTICS["soft_delta"]["note"]


def test_normalized_move_row_divides_by_before_metrics() -> None:
    row = {
        "case_id": 1,
        "move_type": "simple_compaction",
        "before_metrics": {"hpwl_proxy": 10.0, "bbox_area": 200.0},
        "hpwl_delta": 2.0,
        "bbox_delta": -20.0,
        "boundary_delta": 0.1,
        "soft_delta": -1,
    }

    normalized = normalized_move_row(row)

    assert normalized["hpwl_delta_norm"] == 0.2
    assert normalized["bbox_delta_norm"] == -0.1


def test_layout_pathology_metrics_detect_balance_and_extreme_aspect() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    placements = {
        0: (0.0, 0.0, 10.0, 1.0),
        1: (12.0, 0.0, 2.0, 2.0),
        2: (14.0, 0.0, 2.0, 2.0),
        3: (16.0, 0.0, 2.0, 2.0),
    }
    frame = PuzzleFrame(0.0, 0.0, 25.0, 10.0, density=0.9, variant="unit")

    metrics = layout_pathology_metrics(case, placements, frame)

    assert 0.0 < metrics["occupancy_ratio"] <= 1.0
    assert metrics["extreme_aspect_count"] == 1
    assert "quadrant_entropy" in metrics


def test_label_case_pathology_marks_expensive_compaction() -> None:
    labels = label_case_pathology(
        {
            "selected_move_type": "simple_compaction",
            "boundary_delta": 0.1,
            "hpwl_delta": 5.0,
            "bbox_delta": -20.0,
            "soft_delta": -1,
        },
        {"left_right_balance": 0.1, "top_bottom_balance": 0.1, "extreme_aspect_count": 0},
        {"left_right_balance": 0.3, "top_bottom_balance": 0.1, "extreme_aspect_count": 0},
        hpwl_delta_norm=0.5,
        bbox_delta_norm=-0.1,
    )

    assert "compaction_boundary_gain_too_expensive" in labels
    assert "spatial_imbalance_after_compaction" in labels
    assert "hpwl_regression" in labels


def test_scale_coverage_reports_large_xl_gap_and_guard_candidates() -> None:
    profiles = [
        {"case_id": 0, "size_bucket": "small"},
        {"case_id": 1, "size_bucket": "medium"},
    ]
    selections = [
        {"case_id": 0, "selected_move_type": "original", "boundary_delta": 0.0},
        {"case_id": 1, "selected_move_type": "simple_compaction", "boundary_delta": 0.1},
    ]
    pathology = [
        {
            "case_id": 0,
            "pathology_labels": ["good_original"],
            "hpwl_delta_norm": 0.0,
            "bbox_delta_norm": 0.0,
            "pathology_delta": {"left_right_balance_delta": 0.0, "top_bottom_balance_delta": 0.0},
        },
        {
            "case_id": 1,
            "selected_move_type": "simple_compaction",
            "pathology_labels": ["hpwl_regression"],
            "hpwl_delta_norm": 0.6,
            "bbox_delta_norm": 0.0,
            "soft_delta": -1.0,
            "hpwl_regression_per_boundary_gain": 50.0,
            "pathology_delta": {"left_right_balance_delta": 0.2, "top_bottom_balance_delta": 0.0},
            "suspicious": True,
        },
    ]

    coverage = scale_coverage_report(profiles, selections, pathology)
    guards = guard_calibration_candidates(pathology)

    assert coverage["buckets"]["large"]["coverage_gap"] is True
    assert coverage["buckets"]["xl"]["coverage_gap"] is True
    assert guards["suspicious_case_ids"] == [1]
