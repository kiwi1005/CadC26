from __future__ import annotations

from puzzleplace.experiments.step7t_phase4_review import review_step7t_phase4


def test_step7t_phase4_review_passes_sidecar_but_blocks_runtime() -> None:
    active_summary = {
        "meaningful_cost_eps": 1e-7,
        "phase4_gate_open": True,
        "candidate_count": 400,
        "strict_winner_count": 3,
        "strict_winner_case_count": 3,
        "per_case": [
            {"case_id": 24, "blocker": "strict_active_soft_repair_found"},
            {"case_id": 51, "blocker": "strict_active_soft_repair_found"},
            {"case_id": 76, "blocker": "strict_active_soft_repair_found"},
        ],
    }
    visual = {
        "decision": "strict_winner_visual_sanity_pass",
        "exact_strict_winner_count": 3,
        "records": [
            {
                "case_id": case_id,
                "candidate_id": f"c{case_id}",
                "hard_feasible": True,
                "all_vector_nonregressing": True,
                "strict_meaningful_winner": True,
                "delta_exact": {
                    "official_like_cost_delta": -0.01,
                    "hpwl_delta": -0.001,
                    "bbox_area_delta": 0.0,
                    "soft_constraint_delta": -0.01,
                },
                "stored_delta": {
                    "official_like_cost_delta": -0.01,
                    "hpwl_delta": -0.001,
                    "bbox_area_delta": 0.0,
                    "soft_constraint_delta": -0.01,
                },
                "stored_vs_exact_max_abs_delta_error": 0.0,
            }
            for case_id in (24, 51, 76)
        ],
    }

    review = review_step7t_phase4(active_summary, visual)

    assert review["decision"] == "phase4_review_pass_runtime_adapter_required"
    assert review["sidecar_phase4_review_pass"] is True
    assert review["runtime_integration_gate_open"] is False
    assert review["strict_winner_cases"] == [24, 51, 76]
    assert "live_layout_adapter_missing" in review["runtime_blockers"]


def test_step7t_phase4_review_fails_with_only_two_cases() -> None:
    active_summary = {
        "meaningful_cost_eps": 1e-7,
        "phase4_gate_open": True,
        "candidate_count": 200,
        "strict_winner_count": 2,
        "strict_winner_case_count": 2,
        "per_case": [],
    }
    visual = {
        "decision": "strict_winner_visual_sanity_pass",
        "exact_strict_winner_count": 2,
        "records": [
            {
                "case_id": case_id,
                "candidate_id": f"c{case_id}",
                "hard_feasible": True,
                "all_vector_nonregressing": True,
                "strict_meaningful_winner": True,
                "delta_exact": {
                    "official_like_cost_delta": -0.01,
                    "hpwl_delta": -0.001,
                    "bbox_area_delta": 0.0,
                    "soft_constraint_delta": -0.01,
                },
                "stored_delta": {
                    "official_like_cost_delta": -0.01,
                    "hpwl_delta": -0.001,
                    "bbox_area_delta": 0.0,
                    "soft_constraint_delta": -0.01,
                },
                "stored_vs_exact_max_abs_delta_error": 0.0,
            }
            for case_id in (24, 51)
        ],
    }

    review = review_step7t_phase4(active_summary, visual)

    assert review["decision"] == "phase4_review_failed"
    assert review["sidecar_phase4_review_pass"] is False
