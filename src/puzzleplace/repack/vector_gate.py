"""Vector objective gates for Step7P synthetic causal repacker."""

from __future__ import annotations

from typing import Any

EPS = 1e-9


def vector_nonregressing(vector: dict[str, float]) -> bool:
    """Require HPWL/BBox/Soft to be non-regressing independently."""

    return (
        float(vector.get("hpwl_delta", 0.0)) <= EPS
        and float(vector.get("bbox_area_delta", 0.0)) <= EPS
        and float(vector.get("soft_constraint_delta", 0.0)) <= EPS
    )


def reject_reason(vector: dict[str, float]) -> str | None:
    if float(vector.get("soft_constraint_delta", 0.0)) > EPS:
        return "reject_soft_regression"
    if float(vector.get("bbox_area_delta", 0.0)) > EPS:
        return "reject_bbox_regression"
    if float(vector.get("hpwl_delta", 0.0)) > EPS:
        return "reject_hpwl_regression"
    return None


def pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated candidates under HPWL/BBox/Soft deltas."""

    accepted = [row for row in candidates if row.get("accepted")]
    front: list[dict[str, Any]] = []
    for candidate in accepted:
        vector = objective_vector(candidate)
        dominated = False
        for other in accepted:
            if other is candidate:
                continue
            other_vector = objective_vector(other)
            if dominates(other_vector, vector):
                dominated = True
                break
        if not dominated:
            front.append({**candidate, "pareto_front": True})
    return front


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = ("hpwl_delta", "bbox_area_delta", "soft_constraint_delta")
    return all(a[key] <= b[key] + EPS for key in keys) and any(
        a[key] < b[key] - EPS for key in keys
    )


def objective_vector(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("objective_vector")
    vector = raw if isinstance(raw, dict) else candidate
    return {
        "hpwl_delta": float(vector.get("hpwl_delta", 0.0)),
        "bbox_area_delta": float(vector.get("bbox_area_delta", 0.0)),
        "soft_constraint_delta": float(vector.get("soft_constraint_delta", 0.0)),
    }
