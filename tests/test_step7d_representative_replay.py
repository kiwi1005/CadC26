from __future__ import annotations

from puzzleplace.diagnostics.case_profile import build_case_profile, profile_summary_by_bucket
from puzzleplace.experiments.representative_suite import select_representative_suite
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_case_profile_emits_required_fields_and_labels() -> None:
    case = make_step6g_synthetic_case(case_id=0, block_count=24)
    placements = {idx: (float(idx) * 3.0, 0.0, 10.0, 1.0) for idx in range(24)}
    frame = PuzzleFrame(0.0, 0.0, 100.0, 20.0, density=0.8, variant="unit")

    profile = build_case_profile(case, placements, frame, candidate_family_usage={"fallback": 1})

    assert profile["block_count"] == 24
    assert "terminal_count" in profile
    assert profile["candidate_family_summary"]["fallback"] == 1
    assert "aspect-heavy" in profile["pathology_labels"]


def test_representative_suite_selects_expected_categories() -> None:
    profiles = [
        {
            "case_id": 1,
            "size_bucket": "small",
            "boundary_failure_rate": 0.1,
            "extreme_aspect_area_fraction": 0.0,
            "block_count": 24,
            "pathology_labels": [],
        },
        {
            "case_id": 2,
            "size_bucket": "small",
            "boundary_failure_rate": 0.8,
            "extreme_aspect_area_fraction": 0.4,
            "block_count": 25,
            "pathology_labels": ["aspect-heavy"],
        },
        {
            "case_id": 70,
            "size_bucket": "large",
            "boundary_failure_rate": 0.7,
            "extreme_aspect_area_fraction": 0.1,
            "mib_count": 0,
            "grouping_count": 0,
            "block_count": 91,
            "pathology_labels": ["boundary-heavy"],
        },
        {
            "case_id": 95,
            "size_bucket": "xl",
            "boundary_failure_rate": 0.2,
            "extreme_aspect_area_fraction": 0.5,
            "area_utilization_proxy": 0.2,
            "hole_fragmentation_proxy": 0.4,
            "block_count": 116,
            "pathology_labels": ["sparse", "fragmented"],
        },
    ]

    suite = select_representative_suite(profiles)

    assert "small-good" in suite["coverage"]["covered_categories"]
    assert "small-bad" in suite["coverage"]["covered_categories"]
    assert suite["coverage"]["has_xl"] is True


def test_profile_summary_by_bucket_includes_large_xl_keys() -> None:
    summary = profile_summary_by_bucket(
        [
            {
                "size_bucket": "large",
                "extreme_aspect_count": 2,
                "extreme_aspect_area_fraction": 0.1,
                "boundary_failure_rate": 0.2,
                "hole_fragmentation_proxy": 0.3,
            },
            {
                "size_bucket": "xl",
                "extreme_aspect_count": 4,
                "extreme_aspect_area_fraction": 0.2,
                "boundary_failure_rate": 0.4,
                "hole_fragmentation_proxy": 0.5,
            },
        ]
    )

    assert summary["large"]["case_count"] == 1
    assert summary["xl"]["case_count"] == 1
