from __future__ import annotations

from collections import defaultdict
from typing import Any

REQUIRED_CATEGORIES = (
    "small-good",
    "small-bad",
    "medium-good",
    "medium-aspect-bad",
    "large-boundary-bad",
    "large-aspect-bad",
    "large-MIB/group-heavy",
    "XL-sparse",
    "XL-fragmented",
)


def suite_category_candidates(profiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        bucket = str(profile.get("size_bucket"))
        labels = set(profile.get("pathology_labels", []))
        if bucket == "small":
            if profile["boundary_failure_rate"] < 0.5 and "aspect-heavy" not in labels:
                buckets["small-good"].append(profile)
            else:
                buckets["small-bad"].append(profile)
        elif bucket == "medium":
            if "aspect-heavy" in labels:
                buckets["medium-aspect-bad"].append(profile)
            elif profile["boundary_failure_rate"] < 0.5:
                buckets["medium-good"].append(profile)
        elif bucket == "large":
            if "boundary-heavy" in labels:
                buckets["large-boundary-bad"].append(profile)
            if "aspect-heavy" in labels:
                buckets["large-aspect-bad"].append(profile)
            if "MIB/group-heavy" in labels:
                buckets["large-MIB/group-heavy"].append(profile)
        elif bucket == "xl":
            if "sparse" in labels:
                buckets["XL-sparse"].append(profile)
            if "fragmented" in labels or "aspect-heavy" in labels:
                buckets["XL-fragmented"].append(profile)
    return {key: rows for key, rows in buckets.items()}


def select_representative_suite(
    profiles: list[dict[str, Any]],
    *,
    max_per_category: int = 1,
) -> dict[str, Any]:
    candidates = suite_category_candidates(profiles)
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for category in REQUIRED_CATEGORIES:
        rows = candidates.get(category, []) or fallback_category_candidates(category, profiles)
        ranked = sorted(rows, key=lambda row: category_rank_key(category, row), reverse=True)
        if not ranked:
            missing.append(category)
            continue
        for row in ranked[:max_per_category]:
            selected.append({"category": category, **row})
    merged: dict[int, dict[str, Any]] = {}
    for row in selected:
        case_id = int(row["case_id"])
        category = str(row["category"])
        if case_id not in merged:
            merged[case_id] = {**row, "categories": [category]}
        elif category not in merged[case_id]["categories"]:
            merged[case_id]["categories"].append(category)
    deduped = list(merged.values())
    coverage = {
        "required_categories": list(REQUIRED_CATEGORIES),
        "covered_categories": sorted(
            {category for row in deduped for category in row.get("categories", [row["category"]])}
        ),
        "missing_categories": missing,
        "selected_case_ids": [row["case_id"] for row in deduped],
        "has_large": any(row.get("size_bucket") == "large" for row in deduped),
        "has_xl": any(row.get("size_bucket") == "xl" for row in deduped),
    }
    return {"selected_cases": deduped, "coverage": coverage}


def category_rank_key(category: str, row: dict[str, Any]) -> tuple[float, float, float]:
    if "good" in category:
        return (
            -float(row.get("boundary_failure_rate", 0.0)),
            -float(row.get("extreme_aspect_area_fraction", 0.0)),
            float(row.get("block_count", 0.0)),
        )
    if "aspect" in category or "fragmented" in category:
        return (
            float(row.get("extreme_aspect_area_fraction", 0.0)),
            float(row.get("hole_fragmentation_proxy", 0.0)),
            float(row.get("block_count", 0.0)),
        )
    if "boundary" in category:
        return (
            float(row.get("boundary_failure_rate", 0.0)),
            float(row.get("block_count", 0.0)),
            float(row.get("extreme_aspect_area_fraction", 0.0)),
        )
    if "MIB/group" in category:
        return (
            float(row.get("mib_count", 0.0)) + float(row.get("grouping_count", 0.0)),
            float(row.get("block_count", 0.0)),
            float(row.get("boundary_failure_rate", 0.0)),
        )
    if "sparse" in category:
        return (
            -float(row.get("area_utilization_proxy", 0.0)),
            float(row.get("block_count", 0.0)),
            float(row.get("hole_fragmentation_proxy", 0.0)),
        )
    return (float(row.get("block_count", 0.0)), 0.0, 0.0)


def fallback_category_candidates(
    category: str, profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if category.startswith("small"):
        return [row for row in profiles if row.get("size_bucket") == "small"]
    if category.startswith("medium"):
        return [row for row in profiles if row.get("size_bucket") == "medium"]
    if category.startswith("large"):
        return [row for row in profiles if row.get("size_bucket") == "large"]
    if category.startswith("XL"):
        return [row for row in profiles if row.get("size_bucket") == "xl"]
    return []
