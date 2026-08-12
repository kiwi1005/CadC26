"""Small baseline-aware score and family-routing helpers.

These helpers are intentionally independent of the candidate generator.  The
runtime can use them as a cheap score proxy while keeping the exact verifier as
the final authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from hcfp.verify import ALPHA, BETA, GAMMA, INFEASIBLE_COST


@dataclass(frozen=True)
class CandidateMetrics:
    """Runtime-visible metrics needed for approximate score reconstruction."""

    layout_area: float
    hpwl: float
    violations_relative: float
    hard_feasible: bool = True


@dataclass(frozen=True)
class BaselineInterval:
    """Conservative interval for a case baseline.

    The lower bounds make a challenger look worse; the upper bounds make an
    incumbent look better.  This is the pair used by
    :func:`conservative_promotion`.
    """

    area_low: float
    area_high: float
    hpwl_low: float
    hpwl_high: float

    def __post_init__(self) -> None:
        area_low = _positive("area_low", self.area_low)
        area_high = _positive("area_high", self.area_high)
        hpwl_low = _positive("hpwl_low", self.hpwl_low)
        hpwl_high = _positive("hpwl_high", self.hpwl_high)
        if area_low > area_high:
            raise ValueError("area_low must not exceed area_high")
        if hpwl_low > hpwl_high:
            raise ValueError("hpwl_low must not exceed hpwl_high")

    @classmethod
    def exact(cls, *, area: float, hpwl: float) -> "BaselineInterval":
        """Create a zero-width interval for a known baseline."""

        return cls(area, area, hpwl, hpwl)


@dataclass(frozen=True)
class LocalCost:
    """Approximate local official-v10 score and its useful components."""

    cost: float
    uncapped_cost: float
    hpwl_gap: float
    area_gap: float
    quality_factor: float
    runtime_term: float


def estimate_local_cost(
    candidate: CandidateMetrics,
    *,
    baseline_area: float,
    baseline_hpwl: float,
    runtime_factor: float = 1.0,
) -> LocalCost:
    """Reconstruct the pinned local official cost from runtime metrics.

    ``baseline_area`` and ``baseline_hpwl`` are predictions when this helper
    is used by the router.  The exact scorer remains authoritative after a
    candidate is selected.
    """

    layout_area = _positive("layout_area", candidate.layout_area)
    hpwl = _nonnegative("hpwl", candidate.hpwl)
    violations = _nonnegative("violations_relative", candidate.violations_relative)
    area = _positive("baseline_area", baseline_area)
    baseline_wire = _positive("baseline_hpwl", baseline_hpwl)
    runtime = _nonnegative("runtime_factor", runtime_factor)

    hpwl_gap = (hpwl - baseline_wire) / baseline_wire
    area_gap = (layout_area - area) / area
    quality_factor = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    runtime_term = max(0.7, math.pow(max(0.01, runtime), GAMMA))
    uncapped = quality_factor * math.exp(BETA * violations) * runtime_term
    cost = (
        INFEASIBLE_COST
        if not candidate.hard_feasible
        else min(uncapped, INFEASIBLE_COST - 1.0e-6)
    )
    return LocalCost(cost, uncapped, hpwl_gap, area_gap, quality_factor, runtime_term)


def approximate_local_cost(
    candidate: CandidateMetrics,
    *,
    baseline_area: float,
    baseline_hpwl: float,
    runtime_factor: float = 1.0,
) -> float:
    """Return only the capped local score for lightweight call sites."""

    return estimate_local_cost(
        candidate,
        baseline_area=baseline_area,
        baseline_hpwl=baseline_hpwl,
        runtime_factor=runtime_factor,
    ).cost


def conservative_promotion(
    challenger: CandidateMetrics,
    incumbent: CandidateMetrics,
    baseline: BaselineInterval,
    *,
    challenger_runtime_factor: float = 1.0,
    incumbent_runtime_factor: float = 1.0,
) -> bool:
    """Promote only when the worst challenger beats the best incumbent.

    The challenger is scored with the interval's lower baselines, while the
    incumbent is scored with the upper baselines.  Strict comparison avoids
    replacing a known incumbent on a proxy tie.
    """

    if not challenger.hard_feasible:
        return False
    if not incumbent.hard_feasible:
        return True
    challenger_cost = approximate_local_cost(
        challenger,
        baseline_area=baseline.area_low,
        baseline_hpwl=baseline.hpwl_low,
        runtime_factor=challenger_runtime_factor,
    )
    incumbent_cost = approximate_local_cost(
        incumbent,
        baseline_area=baseline.area_high,
        baseline_hpwl=baseline.hpwl_high,
        runtime_factor=incumbent_runtime_factor,
    )
    return challenger_cost < incumbent_cost


def weighted_family_priority(
    block_count: int,
    expected_cost_reduction: float,
    added_runtime_seconds: float,
    *,
    weight_denominator: float = 12.0,
) -> float:
    """Return ``exp(n / 12) * expected_gain / added_runtime``.

    A non-positive expected gain is never worth allocating family seeds.
    """

    if int(block_count) != block_count or block_count <= 0:
        raise ValueError("block_count must be a positive integer")
    denominator = _positive("weight_denominator", weight_denominator)
    runtime = _positive("added_runtime_seconds", added_runtime_seconds)
    gain = max(0.0, _finite("expected_cost_reduction", expected_cost_reduction))
    return math.exp(float(block_count) / denominator) * gain / runtime


def allocate_family_seeds(
    block_count: int,
    expected_cost_reduction: float,
    added_runtime_seconds: float,
    *,
    two_seed_threshold: float,
    four_seed_threshold: float,
    weight_denominator: float = 12.0,
) -> int:
    """Allocate 0, 2, or 4 seeds using an explicit benefit/runtime policy."""

    two = _nonnegative("two_seed_threshold", two_seed_threshold)
    four = _nonnegative("four_seed_threshold", four_seed_threshold)
    if two > four:
        raise ValueError("two_seed_threshold must not exceed four_seed_threshold")
    priority = weighted_family_priority(
        block_count,
        expected_cost_reduction,
        added_runtime_seconds,
        weight_denominator=weight_denominator,
    )
    if priority < two:
        return 0
    if priority < four:
        return 2
    return 4


def route_family_seeds(
    block_count: int,
    expected_cost_reductions: Mapping[str, float],
    added_runtime_seconds: Mapping[str, float],
    *,
    two_seed_threshold: float,
    four_seed_threshold: float,
    weight_denominator: float = 12.0,
) -> dict[str, int]:
    """Allocate seed counts for several named families with one policy."""

    if set(expected_cost_reductions) != set(added_runtime_seconds):
        raise ValueError("family gain/runtime keys must match")
    return {
        family: allocate_family_seeds(
            block_count,
            expected_cost_reductions[family],
            added_runtime_seconds[family],
            two_seed_threshold=two_seed_threshold,
            four_seed_threshold=four_seed_threshold,
            weight_denominator=weight_denominator,
        )
        for family in sorted(expected_cost_reductions)
    }


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result
