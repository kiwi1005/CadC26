from pathlib import Path

from puzzleplace.search.quality_filtered_macro_integration import (
    build_archive,
    classify_row,
    run_step7n_i,
)


def test_classify_archive_and_recovery_rows() -> None:
    assert (
        classify_row(
            {
                "step7n_h_c_retained": True,
                "step7n_h_c_retain_reasons": ["official_like_winner_preservation"],
            }
        )
        == "archive_candidate"
    )
    assert (
        classify_row(
            {
                "step7n_h_c_retained": True,
                "step7n_h_c_retain_reasons": ["local_starved_case_representative"],
            }
        )
        == "recovery_report_only"
    )
    assert classify_row({"step7n_h_c_retained": False}) == "rejected_by_h_c_filter"


def test_archive_uses_only_archive_candidates() -> None:
    archive = build_archive(
        [
            {
                "case_id": 24,
                "step7n_i_selected_for_archive": True,
                "lane": "macro_aware_slot_matching",
                "source_branch": "step7n_c",
                "official_like_cost_improving": True,
                "dominated_by_original": False,
                "official_like_cost_delta": -1.0,
                "hpwl_delta": -1.0,
                "bbox_area_delta": 0.0,
                "soft_constraint_delta": 0.0,
                "candidate_id": "winner",
                "metric_regression_reason": "none",
                "step7n_i_target_calibration": {},
                "step7n_h_c_retain_reasons": ["official_like_winner_preservation"],
            },
            {"case_id": 24, "step7n_i_selected_for_archive": False},
        ]
    )
    assert archive["archive_candidate_count"] == 1
    assert archive["front_contribution_by_lane"] == {"macro_aware_slot_matching": 1}


def test_step7n_i_real_artifact_integration(tmp_path: Path) -> None:
    result = run_step7n_i(Path("."), tmp_path)
    metrics = result["metrics"]
    assert result["decision"] == "promote_quality_filter_to_step7n_g_sidecar"
    assert metrics["official_like_winner_preservation_count"] == 3
    assert metrics["non_anchor_pareto_preservation_count"] == 5
    assert metrics["dominated_by_original_archive_count"] == 0
    assert (tmp_path / "step7n_i_decision.md").exists()
