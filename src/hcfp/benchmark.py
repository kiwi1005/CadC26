"""Exact-result comparison and promotion reporting for HCFP experiments."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean
from typing import Any


BUCKETS = ((21, 64), (65, 95), (96, 105), (106, 115), (116, 120))
DELTA_FIELDS = ("cost", "hpwl_gap", "area_gap", "violations_relative", "runtime_seconds")
ATTRIBUTION_SOURCES = (
    "fallback",
    "analytic_initial",
    "learned_initial",
    "analytic_relaxed",
    "learned_relaxed",
)
ANALYTIC_SOURCES = frozenset(("fallback", "analytic_initial", "analytic_relaxed"))
LEARNED_SOURCES = frozenset(("learned_initial", "learned_relaxed"))
CANDIDATE_TYPES = ("fallback", "analytic", "learned_residual", "topology")


def candidate_source_layout(population: int, learned_count: int) -> tuple[str, ...]:
    """Return the exact analytic/learned source for each tail candidate index."""

    if population <= 0:
        raise ValueError("population must be positive")
    if learned_count < 0:
        raise ValueError("learned_count must be non-negative")
    return (
        ("fallback",)
        + ("analytic_initial",) * population
        + ("learned_initial",) * learned_count
        + ("analytic_relaxed",) * population
        + ("learned_relaxed",) * learned_count
    )


def uncapped_objective(hpwl_gap: float, area_gap: float, violations_relative: float) -> float:
    """Return the official v10 quality objective without runtime or feasible cap."""

    values = tuple(float(value) for value in (hpwl_gap, area_gap, violations_relative))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("objective inputs must be finite")
    hpwl, area, violations = values
    return (1.0 + 0.5 * (max(0.0, hpwl) + max(0.0, area))) * math.exp(2.0 * violations)


def select_candidate_oracle(
    candidates: list[dict[str, Any]],
    *,
    sources: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Select the lowest-objective feasible candidate with stable index tie-breaking."""

    eligible = [
        row
        for row in candidates
        if bool(row["hard_feasible"]) and (sources is None or str(row["source"]) in sources)
    ]
    if not eligible:
        return None
    winner = min(
        eligible,
        key=lambda row: (float(row["uncapped_objective"]), int(row["candidate_index"])),
    )
    return {
        key: winner[key]
        for key in (
            "candidate_index",
            "source",
            "hard_feasible",
            "hpwl_gap",
            "area_gap",
            "violations_relative",
            "official_capped_cost",
            "uncapped_objective",
        )
    }


def candidate_oracles(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall, lane, and per-source feasible oracles."""

    return {
        "overall": select_candidate_oracle(candidates),
        "analytic": select_candidate_oracle(candidates, sources=ANALYTIC_SOURCES),
        "learned": select_candidate_oracle(candidates, sources=LEARNED_SOURCES),
        "by_source": {
            source: select_candidate_oracle(candidates, sources=frozenset((source,)))
            for source in ATTRIBUTION_SOURCES
        },
    }


def summarize_attribution_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate deterministic raw/post-BDP source attribution for case records."""

    ordered = sorted(cases, key=lambda row: int(row["test_id"]))
    return {
        "cases": len(ordered),
        "raw": _summarize_attribution_stage(ordered, "raw"),
        "post_bdp": _summarize_attribution_stage(ordered, "post_bdp", include_incumbent=True),
    }


def summarize_candidate_types(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate candidate-type oracles without conflating topology and residuals."""

    ordered = sorted(cases, key=lambda row: int(row["test_id"]))
    return {
        stage: _summarize_candidate_type_stage(ordered, stage)
        for stage in ("raw", "post_bdp")
    }


def _summarize_candidate_type_stage(
    cases: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    counts = Counter({candidate_type: 0 for candidate_type in CANDIDATE_TYPES})
    feasible = Counter({candidate_type: 0 for candidate_type in CANDIDATE_TYPES})
    oracle_values: dict[str, list[tuple[int, float]]] = {
        candidate_type: [] for candidate_type in CANDIDATE_TYPES
    }
    gains: list[tuple[int, float]] = []
    topology_better = analytic_better = tied = 0
    for case in cases:
        candidates = case[stage]["candidates"]
        for candidate in candidates:
            candidate_type = str(candidate.get("candidate_type", "unknown"))
            counts[candidate_type] += 1
            feasible[candidate_type] += int(bool(candidate["hard_feasible"]))
        for candidate_type in CANDIDATE_TYPES:
            typed = [
                row
                for row in candidates
                if str(row.get("candidate_type")) == candidate_type
            ]
            oracle = select_candidate_oracle(typed)
            if oracle is not None:
                oracle_values[candidate_type].append(
                    (int(case["block_count"]), float(oracle["uncapped_objective"]))
                )
        analytic = candidate_oracles(candidates)["analytic"]
        topology = select_candidate_oracle(
            [row for row in candidates if row.get("candidate_type") == "topology"]
        )
        if analytic is None or topology is None:
            continue
        gain = float(analytic["uncapped_objective"]) - float(
            topology["uncapped_objective"]
        )
        gains.append((int(case["block_count"]), gain))
        if gain > 1.0e-9:
            topology_better += 1
        elif gain < -1.0e-9:
            analytic_better += 1
        else:
            tied += 1
    return {
        "candidate_count_by_type": dict(counts),
        "hard_feasible_by_type": dict(feasible),
        "oracle_available_cases": {
            candidate_type: len(values)
            for candidate_type, values in oracle_values.items()
        },
        "mean_oracle_objective": {
            candidate_type: fmean(value for _, value in values) if values else None
            for candidate_type, values in oracle_values.items()
        },
        "weighted_mean_oracle_objective": {
            candidate_type: _weighted_value(values)
            for candidate_type, values in oracle_values.items()
        },
        "topology_vs_analytic": {
            "comparable_cases": len(gains),
            "topology_better_cases": topology_better,
            "analytic_better_cases": analytic_better,
            "tied_cases": tied,
            "mean_topology_oracle_gain": (
                fmean(gain for _, gain in gains) if gains else None
            ),
            "weighted_mean_topology_oracle_gain": _weighted_gain(gains),
        },
    }


def _summarize_attribution_stage(
    cases: list[dict[str, Any]],
    stage: str,
    *,
    include_incumbent: bool = False,
) -> dict[str, Any]:
    candidate_counts = Counter({source: 0 for source in ATTRIBUTION_SOURCES})
    feasible_counts = Counter({source: 0 for source in ATTRIBUTION_SOURCES})
    source_oracle_counts = Counter({source: 0 for source in ATTRIBUTION_SOURCES})
    oracle_source_wins = Counter({source: 0 for source in ATTRIBUTION_SOURCES})
    source_objectives: dict[str, list[float]] = {source: [] for source in ATTRIBUTION_SOURCES}
    overall_objectives: list[float] = []
    gains: list[tuple[int, float]] = []
    learned_better = analytic_better = tied = 0
    incumbent_sources = Counter({source: 0 for source in ATTRIBUTION_SOURCES})
    incumbent_misses = 0
    incumbent_infeasible = 0
    incumbent_cases = 0

    for case in cases:
        candidates = case[stage]["candidates"]
        oracles = candidate_oracles(candidates)
        for candidate in candidates:
            source = str(candidate["source"])
            candidate_counts[source] += 1
            feasible_counts[source] += int(bool(candidate["hard_feasible"]))
        overall = oracles["overall"]
        if overall is not None:
            overall_objectives.append(float(overall["uncapped_objective"]))
            oracle_source_wins[str(overall["source"])] += 1
        for source, oracle in oracles["by_source"].items():
            if oracle is not None:
                source_oracle_counts[source] += 1
                source_objectives[source].append(float(oracle["uncapped_objective"]))

        analytic = oracles["analytic"]
        learned = oracles["learned"]
        if analytic is not None and learned is not None:
            gain = float(analytic["uncapped_objective"]) - float(learned["uncapped_objective"])
            gains.append((int(case["block_count"]), gain))
            if gain > 1.0e-9:
                learned_better += 1
            elif gain < -1.0e-9:
                analytic_better += 1
            else:
                tied += 1

        if include_incumbent:
            incumbent = case.get("incumbent")
            if incumbent is not None:
                incumbent_cases += 1
                incumbent_sources[str(incumbent["source"])] += 1
                incumbent_infeasible += int(not bool(incumbent["hard_feasible"]))
                if (
                    overall is not None
                    and (
                        not bool(incumbent["hard_feasible"])
                        or float(incumbent["uncapped_objective"])
                        > float(overall["uncapped_objective"]) + 1.0e-9
                    )
                ):
                    incumbent_misses += 1

    total_candidates = sum(candidate_counts.values())
    feasible_candidates = sum(feasible_counts.values())
    result = {
        "candidate_count": total_candidates,
        "hard_feasible_candidates": feasible_candidates,
        "hard_feasibility_rate": (
            feasible_candidates / total_candidates if total_candidates else 0.0
        ),
        "candidate_count_by_source": dict(candidate_counts),
        "hard_feasible_by_source": dict(feasible_counts),
        "overall_oracle_source_counts": dict(oracle_source_wins),
        "source_oracle_available_cases": dict(source_oracle_counts),
        "mean_overall_oracle_objective": fmean(overall_objectives) if overall_objectives else None,
        "mean_source_oracle_objective": {
            source: fmean(values) if values else None for source, values in source_objectives.items()
        },
        "learned_vs_analytic": {
            "comparable_cases": len(gains),
            "learned_better_cases": learned_better,
            "analytic_better_cases": analytic_better,
            "tied_cases": tied,
            "mean_learned_oracle_gain": fmean(gain for _, gain in gains) if gains else None,
            "weighted_mean_learned_oracle_gain": _weighted_gain(gains),
        },
    }
    if include_incumbent:
        result.update(
            {
                "incumbent_evaluated_cases": incumbent_cases,
                "incumbent_infeasible_count": incumbent_infeasible,
                "incumbent_miss_count": incumbent_misses,
                "incumbent_source_counts": dict(incumbent_sources),
            }
        )
    return result


def _weighted_gain(gains: list[tuple[int, float]]) -> float | None:
    if not gains:
        return None
    max_blocks = max(blocks for blocks, _ in gains)
    weights = [math.exp((blocks - max_blocks) / 12.0) for blocks, _ in gains]
    return sum(gain * weight for (_, gain), weight in zip(gains, weights)) / sum(weights)


def _weighted_value(values: list[tuple[int, float]]) -> float | None:
    if not values:
        return None
    max_blocks = max(blocks for blocks, _ in values)
    weights = [math.exp((blocks - max_blocks) / 12.0) for blocks, _ in values]
    return sum(value * weight for (_, value), weight in zip(values, weights)) / sum(
        weights
    )


def weighted_score(rows: list[dict[str, Any]]) -> float:
    """Return the official block-count-weighted cost for result rows."""

    if not rows:
        return 0.0
    weights = [math.exp(int(row["block_count"]) / 12.0) for row in rows]
    return sum(float(row["cost"]) * weight for row, weight in zip(rows, weights)) / sum(weights)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = sorted(float(row["runtime_seconds"]) for row in rows)
    feasible = sum(bool(row["is_feasible"]) for row in rows)
    return {
        "cases": len(rows),
        "feasible": feasible,
        "hard_feasibility_rate": feasible / len(rows) if rows else 0.0,
        "weighted_cost": weighted_score(rows),
        "average_cost": fmean(float(row["cost"]) for row in rows) if rows else 0.0,
        "capped_feasible_cases": sum(classify(row) == "capped_feasible" for row in rows),
        "runtime_p50": percentile(runtimes, 0.50),
        "runtime_p95": percentile(runtimes, 0.95),
        "runtime_p99": percentile(runtimes, 0.99),
        "runtime_max": max(runtimes, default=0.0),
    }


def build_report(
    lanes: dict[str, list[dict[str, Any]]],
    *,
    baseline: str,
    provenance: dict[str, Any] | None = None,
    case_metadata: dict[str, Any] | None = None,
    lane_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one stable comparison report from evaluator-compatible rows."""

    if baseline not in lanes:
        raise ValueError(f"baseline lane {baseline!r} is missing")
    normalized = {name: _ordered_rows(rows) for name, rows in lanes.items()}
    expected = [int(row["test_id"]) for row in normalized[baseline]]
    for name, rows in normalized.items():
        if [int(row["test_id"]) for row in rows] != expected:
            raise ValueError(f"lane {name!r} does not contain the same test ids as baseline")

    lane_summary = {name: summarize(rows) for name, rows in normalized.items()}
    bucket_summary = {
        name: {
            f"{low}-{high}": summarize(
                [row for row in rows if low <= int(row["block_count"]) <= high]
            )
            for low, high in BUCKETS
        }
        for name, rows in normalized.items()
    }
    large_case_summary = {
        name: {
            "106-120": summarize([row for row in rows if 106 <= int(row["block_count"]) <= 120]),
            "116-120": summarize([row for row in rows if 116 <= int(row["block_count"]) <= 120]),
        }
        for name, rows in normalized.items()
    }
    comparisons = {
        name: _compare(normalized[baseline], rows)
        for name, rows in normalized.items()
        if name != baseline
    }
    decisions = {
        name: promotion_decision(
            lane_summary[baseline],
            summary,
            baseline_large=large_case_summary[baseline]["106-120"],
            candidate_large=large_case_summary[name]["106-120"],
        )
        for name, summary in lane_summary.items()
        if name != baseline
    }
    return {
        "schema_version": 1,
        "provenance": provenance or {},
        "case_metadata": case_metadata or {},
        "lane_metadata": lane_metadata or {},
        "baseline": baseline,
        "lanes": normalized,
        "lane_summary": lane_summary,
        "bucket_summary": bucket_summary,
        "large_case_summary": large_case_summary,
        "comparisons": comparisons,
        "promotion_decisions": decisions,
    }


def classify(row: dict[str, Any]) -> str:
    if not bool(row["is_feasible"]):
        return "infeasible"
    return "capped_feasible" if float(row["cost"]) >= 9.99 else "competitive"


def promotion_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_large: dict[str, Any] | None = None,
    candidate_large: dict[str, Any] | None = None,
) -> str:
    if candidate["hard_feasibility_rate"] < 1.0:
        return "REJECT"
    if candidate["capped_feasible_cases"] > baseline["capped_feasible_cases"]:
        return "REJECT"
    if candidate["capped_feasible_cases"] == candidate["cases"]:
        return "HOLD"
    if candidate["weighted_cost"] >= baseline["weighted_cost"]:
        return "HOLD"
    if baseline_large and candidate_large and baseline_large["cases"]:
        if candidate_large["weighted_cost"] > baseline_large["weighted_cost"] + 1.0e-9:
            return "HOLD"
    if candidate["runtime_p50"] > baseline["runtime_p50"] + 1.0e-9:
        return "HOLD"
    if candidate["runtime_p95"] > 1.10 * baseline["runtime_p95"] + 1.0e-9:
        return "HOLD"
    return "PROMOTE"


def percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "test_id",
        "block_count",
        "is_feasible",
        "cost",
        "hpwl_gap",
        "area_gap",
        "violations_relative",
        "runtime_seconds",
    }
    out = []
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"result row is missing {sorted(missing)}")
        item = dict(row)
        item["classification"] = classify(item)
        out.append(item)
    return sorted(out, key=lambda row: int(row["test_id"]))


def _compare(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = []
    for base, current in zip(baseline, candidate):
        row = {"test_id": int(base["test_id"]), "block_count": int(base["block_count"])}
        row.update({field: float(current[field]) - float(base[field]) for field in DELTA_FIELDS})
        deltas.append(row)
    improved = sum(row["cost"] < -1.0e-9 for row in deltas)
    regressed = sum(row["cost"] > 1.0e-9 for row in deltas)
    return {
        "weighted_cost_delta": weighted_score(candidate) - weighted_score(baseline),
        "improved_cases": improved,
        "regressed_cases": regressed,
        "tied_cases": len(deltas) - improved - regressed,
        "per_case_delta": deltas,
    }
