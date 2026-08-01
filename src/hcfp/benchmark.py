"""Exact-result comparison and promotion reporting for HCFP experiments."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any


BUCKETS = ((21, 64), (65, 95), (96, 105), (106, 115), (116, 120))
DELTA_FIELDS = ("cost", "hpwl_gap", "area_gap", "violations_relative", "runtime_seconds")


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
