from __future__ import annotations

import math

import pytest

from hcfp.benchmark import (
    candidate_oracles,
    candidate_source_layout,
    summarize_candidate_types,
    summarize_attribution_cases,
    uncapped_objective,
)


def _candidate(
    index: int,
    source: str,
    objective: float,
    *,
    feasible: bool = True,
    candidate_type: str | None = None,
):
    result = {
        "candidate_index": index,
        "source": source,
        "hard_feasible": feasible,
        "hpwl_gap": objective - 1.0,
        "area_gap": 0.0,
        "violations_relative": 0.0,
        "official_capped_cost": min(objective, 9.999999) if feasible else 10.0,
        "uncapped_objective": objective,
    }
    if candidate_type is not None:
        result["candidate_type"] = candidate_type
    return result


def test_candidate_source_layout_tracks_pruned_learned_population() -> None:
    assert candidate_source_layout(2, 1) == (
        "fallback",
        "analytic_initial",
        "analytic_initial",
        "learned_initial",
        "analytic_relaxed",
        "analytic_relaxed",
        "learned_relaxed",
    )
    with pytest.raises(ValueError, match="population"):
        candidate_source_layout(0, 1)
    assert candidate_source_layout(1, 2) == (
        "fallback",
        "analytic_initial",
        "learned_initial",
        "learned_initial",
        "analytic_relaxed",
        "learned_relaxed",
        "learned_relaxed",
    )


def test_uncapped_objective_matches_v10_quality_without_runtime_or_cap() -> None:
    expected = (1.0 + 0.5 * (2.0 + 0.0)) * math.exp(0.5)
    assert uncapped_objective(2.0, -3.0, 0.25) == pytest.approx(expected)
    assert uncapped_objective(100.0, 100.0, 1.0) > 9.999999


def test_oracle_excludes_infeasible_candidates() -> None:
    candidates = [
        _candidate(0, "fallback", 3.0),
        _candidate(1, "learned_initial", 0.5, feasible=False),
        _candidate(2, "learned_relaxed", 2.0),
    ]

    oracles = candidate_oracles(candidates)

    assert oracles["overall"]["candidate_index"] == 2
    assert oracles["learned"]["candidate_index"] == 2
    assert oracles["by_source"]["learned_initial"] is None


def test_attribution_summary_is_deterministic_and_counts_incumbent_misses() -> None:
    cases = [
        {
            "test_id": 9,
            "block_count": 120,
            "raw": {
                "candidates": [
                    _candidate(0, "fallback", 3.0),
                    _candidate(1, "learned_initial", 2.0),
                ]
            },
            "post_bdp": {
                "candidates": [
                    _candidate(0, "fallback", 1.0),
                    _candidate(1, "learned_initial", 2.0),
                ]
            },
            "incumbent": _candidate(1, "learned_initial", 2.0),
        },
        {
            "test_id": 2,
            "block_count": 32,
            "raw": {
                "candidates": [
                    _candidate(0, "fallback", 1.5),
                    _candidate(1, "learned_initial", 1.5),
                ]
            },
            "post_bdp": {
                "candidates": [
                    _candidate(0, "fallback", 1.5),
                    _candidate(1, "learned_initial", 1.5),
                ]
            },
            "incumbent": _candidate(0, "fallback", 1.5),
        },
        {
            "test_id": 7,
            "block_count": 64,
            "raw": {
                "candidates": [
                    _candidate(0, "fallback", 2.0),
                    _candidate(1, "learned_initial", 1.0, feasible=False),
                ]
            },
            "post_bdp": {
                "candidates": [
                    _candidate(0, "fallback", 2.0),
                    _candidate(1, "learned_initial", 1.0, feasible=False),
                ]
            },
            "incumbent": _candidate(1, "learned_initial", 1.0, feasible=False),
        },
    ]

    summary = summarize_attribution_cases(cases)

    assert summary == summarize_attribution_cases(list(reversed(cases)))
    assert summary["raw"]["overall_oracle_source_counts"]["learned_initial"] == 1
    assert summary["raw"]["learned_vs_analytic"]["learned_better_cases"] == 1
    assert summary["post_bdp"]["incumbent_infeasible_count"] == 1
    assert summary["post_bdp"]["incumbent_miss_count"] == 2
    assert summary["post_bdp"]["incumbent_source_counts"] == {
        "fallback": 1,
        "analytic_initial": 0,
        "learned_initial": 2,
        "analytic_relaxed": 0,
        "learned_relaxed": 0,
    }


def test_candidate_type_summary_separates_topology_from_residuals() -> None:
    candidates = [
        _candidate(0, "fallback", 3.0, candidate_type="fallback"),
        _candidate(1, "analytic_initial", 2.0, candidate_type="analytic"),
        _candidate(
            2,
            "learned_initial",
            1.8,
            candidate_type="learned_residual",
        ),
        _candidate(3, "learned_initial", 1.0, candidate_type="topology"),
    ]
    cases = [
        {
            "test_id": 0,
            "block_count": 120,
            "raw": {"candidates": candidates},
            "post_bdp": {"candidates": candidates},
        }
    ]

    summary = summarize_candidate_types(cases)

    assert summary["raw"]["candidate_count_by_type"]["topology"] == 1
    assert summary["raw"]["oracle_available_cases"]["learned_residual"] == 1
    assert summary["raw"]["mean_oracle_objective"]["topology"] == 1.0
    assert summary["raw"]["topology_vs_analytic"] == {
        "comparable_cases": 1,
        "topology_better_cases": 1,
        "analytic_better_cases": 0,
        "tied_cases": 0,
        "mean_topology_oracle_gain": 1.0,
        "weighted_mean_topology_oracle_gain": 1.0,
    }
