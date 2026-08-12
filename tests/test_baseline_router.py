from __future__ import annotations

import math

import pytest

from hcfp.baseline_router import (
    BaselineInterval,
    CandidateMetrics,
    allocate_family_seeds,
    approximate_local_cost,
    conservative_promotion,
    estimate_local_cost,
    route_family_seeds,
    weighted_family_priority,
)


def test_local_cost_matches_pinned_quality_and_violation_formula() -> None:
    candidate = CandidateMetrics(
        layout_area=120.0,
        hpwl=60.0,
        violations_relative=0.25,
    )
    result = estimate_local_cost(candidate, baseline_area=100.0, baseline_hpwl=50.0)

    expected_quality = 1.0 + 0.5 * (0.2 + 0.2)
    expected_uncapped = expected_quality * math.exp(2.0 * 0.25)
    assert result.hpwl_gap == pytest.approx(0.2)
    assert result.area_gap == pytest.approx(0.2)
    assert result.quality_factor == pytest.approx(expected_quality)
    assert result.uncapped_cost == pytest.approx(expected_uncapped)
    assert result.cost == pytest.approx(expected_uncapped)
    assert approximate_local_cost(
        candidate, baseline_area=100.0, baseline_hpwl=50.0
    ) == pytest.approx(expected_uncapped)


def test_local_cost_caps_infeasible_candidate() -> None:
    candidate = CandidateMetrics(100.0, 50.0, 0.0, hard_feasible=False)
    assert (
        approximate_local_cost(candidate, baseline_area=100.0, baseline_hpwl=50.0)
        == 10.0
    )


def test_conservative_promotion_uses_worst_challenger_and_best_incumbent() -> None:
    interval = BaselineInterval(
        area_low=90.0,
        area_high=110.0,
        hpwl_low=45.0,
        hpwl_high=55.0,
    )
    challenger = CandidateMetrics(100.0, 48.0, 0.0)
    incumbent = CandidateMetrics(100.0, 50.0, 0.0)

    worst_challenger = approximate_local_cost(
        challenger, baseline_area=90.0, baseline_hpwl=45.0
    )
    best_incumbent = approximate_local_cost(
        incumbent, baseline_area=110.0, baseline_hpwl=55.0
    )
    assert worst_challenger > best_incumbent
    assert not conservative_promotion(challenger, incumbent, interval)


def test_conservative_promotion_accepts_clear_feasible_gain() -> None:
    interval = BaselineInterval.exact(area=100.0, hpwl=50.0)
    challenger = CandidateMetrics(95.0, 45.0, 0.0)
    incumbent = CandidateMetrics(110.0, 60.0, 0.1)
    assert conservative_promotion(challenger, incumbent, interval)
    assert not conservative_promotion(
        CandidateMetrics(95.0, 45.0, 0.0, hard_feasible=False), incumbent, interval
    )


def test_seed_priority_weights_large_cases_and_runtime() -> None:
    small = weighted_family_priority(60, 0.1, 1.0)
    large = weighted_family_priority(120, 0.1, 1.0)
    assert large / small == pytest.approx(math.exp(5.0))
    assert weighted_family_priority(120, 0.0, 1.0) == 0.0
    assert weighted_family_priority(120, -1.0, 1.0) == 0.0


def test_seed_allocator_returns_only_zero_two_or_four() -> None:
    assert (
        allocate_family_seeds(
            60,
            0.0,
            1.0,
            two_seed_threshold=1.0,
            four_seed_threshold=2.0,
        )
        == 0
    )
    score = weighted_family_priority(60, 0.1, 1.0)
    assert (
        allocate_family_seeds(
            60,
            0.1,
            1.0,
            two_seed_threshold=score - 0.01,
            four_seed_threshold=score + 0.01,
        )
        == 2
    )
    assert (
        allocate_family_seeds(
            60,
            0.1,
            1.0,
            two_seed_threshold=score - 0.01,
            four_seed_threshold=score,
        )
        == 4
    )


def test_route_family_seeds_requires_matching_inputs() -> None:
    result = route_family_seeds(
        120,
        {"btree": 0.02, "stripe": 0.0},
        {"btree": 1.0, "stripe": 1.0},
        two_seed_threshold=10.0,
        four_seed_threshold=100.0,
    )
    assert set(result) == {"btree", "stripe"}
    assert result["stripe"] == 0
    with pytest.raises(ValueError, match="keys"):
        route_family_seeds(
            120,
            {"btree": 0.02},
            {"stripe": 1.0},
            two_seed_threshold=1.0,
            four_seed_threshold=2.0,
        )
