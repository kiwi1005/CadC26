from __future__ import annotations

from collections import Counter

from puzzleplace.experiments.step7v_live_active_soft_adapter import (
    classify_live_blocker,
    next_recommendation,
)


def test_classify_live_blocker_prefers_infeasible_baseline() -> None:
    assert (
        classify_live_blocker([], False, {"active_violated_boundary_components": [{"x": 1}]})
        == "live_optimizer_baseline_hard_infeasible"
    )


def test_classify_live_blocker_detects_strict_repair() -> None:
    assert (
        classify_live_blocker(
            [{"strict_meaningful_winner": True}],
            True,
            {"active_violated_boundary_components": [{"x": 1}]},
        )
        == "strict_live_active_soft_repair_found"
    )


def test_next_recommendation_for_three_cases_promotes_design_review() -> None:
    assert (
        next_recommendation(3, Counter())
        == "promote_live_active_soft_adapter_to_runtime_integration_design_review"
    )


def test_aggregate_case_summaries_opens_gate_for_three_completed_cases() -> None:
    from puzzleplace.experiments.step7v_live_active_soft_adapter import aggregate_case_summaries

    summary = aggregate_case_summaries([
        {
            "case_id": case_id,
            "status": "completed",
            "strict_winner_count": 1,
            "strict_winner_case_count": 1,
            "per_case": [{"case_id": case_id, "blocker": "strict_live_active_soft_repair_found"}],
            "candidate_rows": [{"case_id": case_id}],
        }
        for case_id in (24, 51, 76)
    ])

    assert summary["decision"] == "live_adapter_phase4_gate_open"
    assert summary["phase4_gate_open"] is True
    assert summary["strict_winner_case_count"] == 3
    assert summary["failed_case_count"] == 0

def test_aggregate_case_summaries_blocks_on_failed_case() -> None:
    from puzzleplace.experiments.step7v_live_active_soft_adapter import aggregate_case_summaries

    summary = aggregate_case_summaries([
        {
            "case_id": 24,
            "status": "completed",
            "strict_winner_count": 3,
            "strict_winner_case_count": 1,
            "per_case": [{"case_id": 24, "blocker": "strict_live_active_soft_repair_found"}],
            "candidate_rows": [],
        },
        {"case_id": 76, "status": "timeout", "blocker": "case_subprocess_timeout"},
    ])

    assert summary["decision"] == "live_adapter_partial_failures_present"
    assert summary["phase4_gate_open"] is False
    assert summary["failed_case_count"] == 1
