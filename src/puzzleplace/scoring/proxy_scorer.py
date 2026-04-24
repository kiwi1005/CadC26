from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from puzzleplace.data import FloorSetCase
from puzzleplace.eval.violation import summarize_violation_profile
from puzzleplace.repair import RepairReport


@dataclass(frozen=True, slots=True)
class ObjectiveCandidate:
    source_id: str
    positions: list[tuple[float, float, float, float]]
    repair_report: RepairReport
    semantic_placed_fraction: float
    semantic_fallback_fraction: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProxyFeatureRow:
    source_id: str
    proxy_hpwl_total: float
    bbox_area: float
    proxy_total_soft_violations: float
    changed_block_fraction: float
    shelf_fallback_count: float
    semantic_fallback_fraction: float
    hard_feasible_after: bool


@dataclass(frozen=True, slots=True)
class ObjectiveSelection:
    candidate: ObjectiveCandidate
    candidate_index: int
    score: float
    features: ProxyFeatureRow
    ranked_indices: list[int]


def _bbox_area(positions: Sequence[tuple[float, float, float, float]]) -> float:
    if not positions:
        return 0.0
    x0 = min(float(x) for x, _y, _w, _h in positions)
    y0 = min(float(y) for _x, y, _w, _h in positions)
    x1 = max(float(x + w) for x, _y, w, _h in positions)
    y1 = max(float(y + h) for _x, y, _w, h in positions)
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _as_placement_dict(
    positions: Sequence[tuple[float, float, float, float]],
) -> dict[int, tuple[float, float, float, float]]:
    return {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(positions)
    }


def proxy_features_for_candidate(
    case: FloorSetCase,
    candidate: ObjectiveCandidate,
) -> ProxyFeatureRow:
    placements = _as_placement_dict(candidate.positions)
    profile = summarize_violation_profile(case, placements)
    block_count = max(case.block_count, 1)
    soft_proxy = (
        float(profile.boundary_distance)
        + float(profile.area_violations)
        + float(profile.dimension_violations)
        + float(profile.overlap_pairs)
    )
    return ProxyFeatureRow(
        source_id=candidate.source_id,
        proxy_hpwl_total=float(profile.connectivity_proxy_cost),
        bbox_area=_bbox_area(candidate.positions),
        proxy_total_soft_violations=soft_proxy,
        changed_block_fraction=float(candidate.repair_report.moved_block_count) / block_count,
        shelf_fallback_count=float(candidate.repair_report.shelf_fallback_count),
        semantic_fallback_fraction=float(candidate.semantic_fallback_fraction),
        hard_feasible_after=bool(candidate.repair_report.hard_feasible_after),
    )


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _value in values]
    return [(value - lo) / (hi - lo) for value in values]


def score_proxy_features(
    features: Sequence[ProxyFeatureRow],
    *,
    scorer_name: str = "hpwl_bbox_soft_repair_proxy",
) -> list[float]:
    if scorer_name == "displacement_proxy":
        return _normalize([row.changed_block_fraction for row in features])
    if scorer_name not in {
        "hpwl_bbox_proxy",
        "hpwl_bbox_soft_proxy",
        "hpwl_bbox_soft_repair_proxy",
    }:
        raise ValueError(f"unknown objective proxy scorer: {scorer_name}")

    hpwl = _normalize([row.proxy_hpwl_total for row in features])
    bbox = _normalize([row.bbox_area for row in features])
    scores = [h + b for h, b in zip(hpwl, bbox, strict=True)]
    if scorer_name in {"hpwl_bbox_soft_proxy", "hpwl_bbox_soft_repair_proxy"}:
        soft = _normalize([row.proxy_total_soft_violations for row in features])
        scores = [score + 0.25 * value for score, value in zip(scores, soft, strict=True)]
    if scorer_name == "hpwl_bbox_soft_repair_proxy":
        changed = _normalize([row.changed_block_fraction for row in features])
        shelf = _normalize([row.shelf_fallback_count for row in features])
        fallback = _normalize([row.semantic_fallback_fraction for row in features])
        scores = [
            score + 0.05 * changed_i + 0.05 * shelf_i + 0.05 * fallback_i
            for score, changed_i, shelf_i, fallback_i in zip(
                scores, changed, shelf, fallback, strict=True
            )
        ]
    return scores


def select_objective_candidate(
    case: FloorSetCase,
    candidates: Sequence[ObjectiveCandidate],
    *,
    scorer_name: str = "hpwl_bbox_soft_repair_proxy",
) -> ObjectiveSelection:
    if not candidates:
        raise ValueError("objective selection requires at least one candidate")
    features = [proxy_features_for_candidate(case, candidate) for candidate in candidates]
    scores = score_proxy_features(features, scorer_name=scorer_name)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: (scores[idx], idx))
    selected_idx = ranked_indices[0]
    return ObjectiveSelection(
        candidate=candidates[selected_idx],
        candidate_index=selected_idx,
        score=float(scores[selected_idx]),
        features=features[selected_idx],
        ranked_indices=ranked_indices,
    )
