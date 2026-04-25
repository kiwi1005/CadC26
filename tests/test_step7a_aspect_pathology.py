from __future__ import annotations

import pytest

from puzzleplace.diagnostics.aspect import (
    abs_log_aspect,
    aspect_by_role,
    aspect_stats,
    case_aspect_pathology,
    correlation_report,
    shape_change_summary,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case


def test_aspect_stats_reports_threshold_counts_and_area_fraction() -> None:
    placements = {
        0: (0.0, 0.0, 1.0, 1.0),
        1: (1.0, 0.0, 10.0, 1.0),
    }

    stats = aspect_stats(placements)

    assert stats["block_count"] == 2
    assert stats["extreme_aspect_count_gt_2_0"] == 1
    assert stats["extreme_aspect_area_fraction_gt_2_0"] == pytest.approx(10.0 / 11.0)
    assert abs_log_aspect(placements[1]) > 2.0


def test_aspect_by_role_marks_boundary_and_mib_buckets() -> None:
    case = make_step6g_synthetic_case(block_count=10)
    placements = {idx: (float(idx), 0.0, 10.0, 1.0) for idx in range(case.block_count)}

    by_role = aspect_by_role(case, placements, threshold=1.0)

    assert by_role["boundary"]["extreme_aspect_count"] >= 1
    assert by_role["MIB"]["extreme_aspect_count"] >= 1


def test_case_pathology_contains_required_step7a_fields() -> None:
    case = make_step6g_synthetic_case(block_count=8)
    pre = {idx: (float(idx), 0.0, 2.0, 2.0) for idx in range(case.block_count)}
    post = dict(pre)
    post[2] = (2.0, 0.0, 9.0, 1.0)

    row = case_aspect_pathology(
        case,
        pre_move_placements=pre,
        post_move_placements=post,
        selected_representative="closest_to_ideal",
        selected_move_type="unit_move",
        candidate_family_usage={"free_rect": 3},
    )

    assert row["selected_representative"] == "closest_to_ideal"
    assert row["extreme_aspect_count"] == 1
    assert row["pre_move_aspect_stats"]["extreme_aspect_count"] == 0
    assert row["shape_changed_by_move"]["unit_move"]["shape_changed_block_ids"] == [2]
    assert row["extreme_aspect_by_candidate_family"] == {"free_rect": 3}


def test_shape_change_summary_detects_dimension_changes_only() -> None:
    before = {0: (0.0, 0.0, 2.0, 2.0), 1: (2.0, 0.0, 2.0, 2.0)}
    after = {0: (1.0, 1.0, 2.0, 2.0), 1: (2.0, 0.0, 4.0, 1.0)}

    summary = shape_change_summary(before, after)

    assert summary["shape_changed_block_ids"] == [1]
    assert summary["shape_changed_count"] == 1


def test_correlation_report_handles_constant_series() -> None:
    rows = [{"case_id": 0, "extreme_aspect_count": 1}, {"case_id": 1, "extreme_aspect_count": 1}]
    metrics = [
        {"case_id": 0, "occupancy_ratio": 0.5},
        {"case_id": 1, "occupancy_ratio": 0.7},
    ]

    report = correlation_report(rows, metrics)

    assert report["extreme_aspect_count_vs_occupancy_ratio"] is None
