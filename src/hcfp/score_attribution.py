"""Exact factor attribution for the pinned official-v10 score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from hcfp.verify import ALPHA, BETA, GAMMA, INFEASIBLE_COST


CAP_LOG = math.log(INFEASIBLE_COST)
FEASIBLE_COST_CAP = INFEASIBLE_COST - 1.0e-6


@dataclass(frozen=True)
class ScoreAttribution:
    """Exact additive and multiplicative components of one official score."""

    hard_feasible: bool
    hpwl_gap: float
    area_gap: float
    quality_gap: float
    quality_factor: float
    quality_contribution: float
    boundary_violations: int | None
    grouping_violations: int | None
    mib_violations: int | None
    total_soft_violations: int | None
    max_possible_violations: int | None
    violations_relative: float
    boundary_contribution: float | None
    grouping_contribution: float | None
    mib_contribution: float | None
    soft_contribution: float
    runtime_factor: float
    runtime_term: float
    runtime_contribution: float
    uncapped_cost: float
    log_uncapped_cost: float
    official_capped_cost: float
    cap_margin: float
    is_capped: bool
    required_soft_fixes_to_uncap: int | None
    soft_fixes_sufficient: bool | None
    required_quality_gap_to_uncap: float | None
    quality_improvement_sufficient: bool
    blocker_classification: str
    soft_breakdown_available: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def attribute_score(
    hpwl_gap: float,
    area_gap: float,
    *,
    boundary_violations: int,
    grouping_violations: int,
    mib_violations: int,
    max_possible_violations: int,
    hard_feasible: bool = True,
    runtime_factor: float = 1.0,
) -> ScoreAttribution:
    """Attribute a score from exact official raw soft-violation counts."""

    boundary = _nonnegative_int("boundary_violations", boundary_violations)
    grouping = _nonnegative_int("grouping_violations", grouping_violations)
    mib = _nonnegative_int("mib_violations", mib_violations)
    maximum = _nonnegative_int("max_possible_violations", max_possible_violations)
    total = boundary + grouping + mib
    if total > maximum:
        raise ValueError("soft violation counts cannot exceed max_possible_violations")
    denominator = max(maximum, 1)
    relative = total / denominator
    return _attribute(
        hpwl_gap,
        area_gap,
        violations_relative=relative,
        hard_feasible=hard_feasible,
        runtime_factor=runtime_factor,
        boundary_violations=boundary,
        grouping_violations=grouping,
        mib_violations=mib,
        total_soft_violations=total,
        max_possible_violations=maximum,
    )


def attribute_score_from_relative(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    *,
    hard_feasible: bool = True,
    runtime_factor: float = 1.0,
    total_soft_violations: int | None = None,
    max_possible_violations: int | None = None,
) -> ScoreAttribution:
    """Attribute legacy rows that expose only normalized soft violations.

    Boundary, grouping, and MIB contributions remain ``None`` because their
    split cannot be reconstructed from a normalized aggregate.
    """

    relative = _finite("violations_relative", violations_relative)
    if not 0.0 <= relative <= 1.0:
        raise ValueError("violations_relative must be between zero and one")
    total = (
        None
        if total_soft_violations is None
        else _nonnegative_int("total_soft_violations", total_soft_violations)
    )
    maximum = (
        None
        if max_possible_violations is None
        else _nonnegative_int("max_possible_violations", max_possible_violations)
    )
    if (total is None) != (maximum is None):
        raise ValueError(
            "total_soft_violations and max_possible_violations must be provided together"
        )
    if total is not None and maximum is not None:
        if total > maximum:
            raise ValueError(
                "total_soft_violations cannot exceed max_possible_violations"
            )
        expected = total / max(maximum, 1)
        if not math.isclose(relative, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("violations_relative is inconsistent with the soft counts")
    return _attribute(
        hpwl_gap,
        area_gap,
        violations_relative=relative,
        hard_feasible=hard_feasible,
        runtime_factor=runtime_factor,
        total_soft_violations=total,
        max_possible_violations=maximum,
    )


def _attribute(
    hpwl_gap: float,
    area_gap: float,
    *,
    violations_relative: float,
    hard_feasible: bool,
    runtime_factor: float,
    boundary_violations: int | None = None,
    grouping_violations: int | None = None,
    mib_violations: int | None = None,
    total_soft_violations: int | None = None,
    max_possible_violations: int | None = None,
) -> ScoreAttribution:
    hpwl = _finite("hpwl_gap", hpwl_gap)
    area = _finite("area_gap", area_gap)
    runtime = _finite("runtime_factor", runtime_factor)
    if runtime < 0.0:
        raise ValueError("runtime_factor must be non-negative")
    if not isinstance(hard_feasible, bool):
        raise TypeError("hard_feasible must be a bool")

    quality_gap = max(0.0, hpwl) + max(0.0, area)
    quality_factor = 1.0 + ALPHA * quality_gap
    quality_contribution = math.log(quality_factor)
    soft_contribution = BETA * violations_relative
    runtime_term = max(0.7, math.pow(max(0.01, runtime), GAMMA))
    runtime_contribution = math.log(runtime_term)
    log_uncapped_cost = quality_contribution + soft_contribution + runtime_contribution
    try:
        uncapped_cost = quality_factor * math.exp(soft_contribution) * runtime_term
    except (
        OverflowError
    ) as exc:  # pragma: no cover - requires pathological finite input.
        raise ValueError("uncapped cost overflowed") from exc
    if not math.isfinite(uncapped_cost):
        raise ValueError("uncapped cost must be finite")

    breakdown = boundary_violations is not None
    denominator = max(max_possible_violations or 0, 1)
    boundary_contribution = (
        BETA * boundary_violations / denominator
        if boundary_violations is not None
        else None
    )
    grouping_contribution = (
        BETA * grouping_violations / denominator
        if grouping_violations is not None
        else None
    )
    mib_contribution = (
        BETA * mib_violations / denominator if mib_violations is not None else None
    )
    if breakdown:
        reconstructed = sum(
            value
            for value in (
                boundary_contribution,
                grouping_contribution,
                mib_contribution,
            )
            if value is not None
        )
        if not math.isclose(
            reconstructed, soft_contribution, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                "soft contributions do not reconstruct violations_relative"
            )

    cap_margin = CAP_LOG - log_uncapped_cost
    is_capped = not hard_feasible or cap_margin <= 0.0
    required_soft, soft_sufficient = _required_soft_fixes(
        hard_feasible,
        log_uncapped_cost,
        total_soft_violations,
        max_possible_violations,
    )
    required_quality, quality_sufficient = _required_quality_improvement(
        hard_feasible,
        log_uncapped_cost,
        quality_gap,
        soft_contribution,
        runtime_contribution,
    )
    blocker = _classify_blocker(
        hard_feasible,
        log_uncapped_cost,
        quality_contribution,
        soft_contribution,
        runtime_contribution,
    )
    official_cost = (
        min(uncapped_cost, FEASIBLE_COST_CAP) if hard_feasible else INFEASIBLE_COST
    )
    return ScoreAttribution(
        hard_feasible=hard_feasible,
        hpwl_gap=hpwl,
        area_gap=area,
        quality_gap=quality_gap,
        quality_factor=quality_factor,
        quality_contribution=quality_contribution,
        boundary_violations=boundary_violations,
        grouping_violations=grouping_violations,
        mib_violations=mib_violations,
        total_soft_violations=total_soft_violations,
        max_possible_violations=max_possible_violations,
        violations_relative=violations_relative,
        boundary_contribution=boundary_contribution,
        grouping_contribution=grouping_contribution,
        mib_contribution=mib_contribution,
        soft_contribution=soft_contribution,
        runtime_factor=runtime,
        runtime_term=runtime_term,
        runtime_contribution=runtime_contribution,
        uncapped_cost=uncapped_cost,
        log_uncapped_cost=log_uncapped_cost,
        official_capped_cost=official_cost,
        cap_margin=cap_margin,
        is_capped=is_capped,
        required_soft_fixes_to_uncap=required_soft,
        soft_fixes_sufficient=soft_sufficient,
        required_quality_gap_to_uncap=required_quality,
        quality_improvement_sufficient=quality_sufficient,
        blocker_classification=blocker,
        soft_breakdown_available=breakdown,
    )


def _required_soft_fixes(
    hard_feasible: bool,
    log_cost: float,
    total: int | None,
    maximum: int | None,
) -> tuple[int | None, bool | None]:
    if not hard_feasible:
        return None, False
    excess = max(0.0, log_cost - CAP_LOG)
    if excess == 0.0:
        return 0, True
    if total is None or maximum is None:
        return None, None
    if maximum == 0:
        return None, False
    needed = math.ceil(excess * maximum / BETA)
    return needed, needed <= total


def _required_quality_improvement(
    hard_feasible: bool,
    log_cost: float,
    quality_gap: float,
    soft_contribution: float,
    runtime_contribution: float,
) -> tuple[float | None, bool]:
    if not hard_feasible:
        return None, False
    if log_cost <= CAP_LOG:
        return 0.0, True
    quality_budget = CAP_LOG - soft_contribution - runtime_contribution
    if quality_budget < 0.0:
        return None, False
    target_factor = math.exp(quality_budget)
    if target_factor < 1.0:
        return None, False
    target_gap = max(0.0, (target_factor - 1.0) / ALPHA)
    return min(quality_gap, max(0.0, quality_gap - target_gap)), True


def _classify_blocker(
    hard_feasible: bool,
    log_cost: float,
    quality_contribution: float,
    soft_contribution: float,
    runtime_contribution: float,
) -> str:
    if not hard_feasible:
        return "hard"
    if log_cost < CAP_LOG:
        return "none"
    soft_fix_uncaps = quality_contribution + runtime_contribution < CAP_LOG
    quality_fix_uncaps = soft_contribution + runtime_contribution < CAP_LOG
    if soft_fix_uncaps and not quality_fix_uncaps:
        return "soft"
    if quality_fix_uncaps and not soft_fix_uncaps:
        return "quality"
    return "mixed"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
