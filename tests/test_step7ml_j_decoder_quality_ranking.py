from __future__ import annotations

from puzzleplace.search.decoder_candidate_quality_ranking import _dominates, select_candidates


def test_dominates_requires_no_worse_and_one_better() -> None:
    a = {
        "feasibility_rank": 0,
        "route_rank": 0,
        "official_like_cost_delta": -1.0,
        "hpwl_delta": 0.0,
        "bbox_area_delta": 0.0,
        "soft_constraint_delta": 0.0,
    }
    b = {**a, "official_like_cost_delta": 0.0}
    assert _dominates(a, b)
    assert not _dominates(b, a)


def test_select_preserves_official_winner_even_if_dominated() -> None:
    vectors = [
        {
            "candidate_id": "c0",
            "case_id": 1,
            "decoder": "slot",
            "hard_feasible_non_noop": True,
            "feasibility_rank": 0,
            "route_rank": 1,
            "official_like_cost_delta": -0.1,
            "hpwl_delta": 0.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "global_report_only": False,
            "metric_regressing": False,
            "official_like_improving": True,
            "quality_gate_pass": True,
        }
    ]
    selected, report = select_candidates(vectors)
    assert selected[0]["candidate_id"] == "c0"
    assert report["rows"][0]["selected"] is True


def test_select_filters_global_report_only() -> None:
    vectors = [
        {
            "candidate_id": "c1",
            "case_id": 1,
            "decoder": "slot",
            "hard_feasible_non_noop": True,
            "feasibility_rank": 0,
            "route_rank": 3,
            "official_like_cost_delta": -1.0,
            "hpwl_delta": -1.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "global_report_only": True,
            "metric_regressing": False,
            "official_like_improving": False,
            "quality_gate_pass": False,
        }
    ]
    selected, _report = select_candidates(vectors)
    assert selected == []
