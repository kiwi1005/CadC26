from __future__ import annotations

import math
from collections import Counter
from typing import Any

from puzzleplace.data import FloorSetCase
from puzzleplace.research.move_library import (
    profile_case,
)
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _area,
    _bbox_from_placements,
    frame_diagnostics,
)

METRIC_SEMANTICS: dict[str, Any] = {
    "boundary_delta": {
        "definition": "after boundary satisfaction rate minus before rate",
        "positive_means": "better",
        "symbol": "b_d",
    },
    "hpwl_delta": {
        "definition": "after HPWL proxy minus before HPWL proxy",
        "positive_means": "worse",
        "normalized": "hpwl_delta / max(before_hpwl_proxy, eps)",
        "symbol": "hpwl_d",
    },
    "bbox_delta": {
        "definition": "after final bbox area minus before final bbox area",
        "positive_means": "worse",
        "normalized": "bbox_delta / max(before_bbox_area, eps)",
        "symbol": "bbox_d",
    },
    "soft_delta": {
        "definition": "after soft boundary violation count minus before count",
        "positive_means": "worse",
        "note": "soft_delta < 0 means fewer violations",
        "symbol": "soft_d",
    },
    "official_proxy_delta": {
        "definition": "not available in Step6M sidecar; report null unless official metric exists",
        "positive_means": "worse",
    },
    "soft_penalty_multiplier_delta": {
        "definition": "exp-style official soft multiplier delta unavailable in Step6M proxy",
        "positive_means": "worse",
    },
}


def normalized_move_row(row: dict[str, Any]) -> dict[str, Any]:
    before = dict(row.get("before_metrics", {}))
    return {
        "case_id": row.get("case_id"),
        "move_type": row.get("move_type"),
        "hpwl_delta_raw": float(row.get("hpwl_delta", 0.0)),
        "hpwl_delta_norm": float(row.get("hpwl_delta", 0.0))
        / max(float(before.get("hpwl_proxy", 0.0)), 1e-6),
        "bbox_delta_raw": float(row.get("bbox_delta", 0.0)),
        "bbox_delta_norm": float(row.get("bbox_delta", 0.0))
        / max(float(before.get("bbox_area", 0.0)), 1e-6),
        "boundary_delta_raw": float(row.get("boundary_delta", 0.0)),
        "soft_delta_raw": float(row.get("soft_delta", 0.0)),
        "official_proxy_delta": None,
        "soft_penalty_multiplier_delta": None,
    }


def layout_pathology_metrics(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    fallback_count: int = 0,
) -> dict[str, Any]:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return {
            "occupancy_ratio": 0.0,
            "frame_utilization": 0.0,
            "left_right_balance": 0.0,
            "top_bottom_balance": 0.0,
            "quadrant_entropy": 0.0,
            "largest_empty_rectangle_ratio": 1.0,
            "placed_centroid_vs_pin_centroid_distance": 0.0,
            "terminal_heavy_mean_distance_to_pin": 0.0,
            "extreme_aspect_count": 0,
            "fallback_count": fallback_count,
        }
    total_rect_area = sum(max(w, 0.0) * max(h, 0.0) for _x, _y, w, h in placements.values())
    bbox_area = max(_area(bbox), 1e-6)
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    left_area = right_area = top_area = bottom_area = 0.0
    quadrant_areas = [0.0, 0.0, 0.0, 0.0]
    weighted_cx = weighted_cy = 0.0
    extreme_aspect = 0
    for box in placements.values():
        x, y, w, h = box
        area = max(w, 0.0) * max(h, 0.0)
        bx, by = x + w / 2.0, y + h / 2.0
        weighted_cx += bx * area
        weighted_cy += by * area
        left_area += area if bx < cx else 0.0
        right_area += area if bx >= cx else 0.0
        bottom_area += area if by < cy else 0.0
        top_area += area if by >= cy else 0.0
        qx = 0 if bx < cx else 1
        qy = 0 if by < cy else 1
        quadrant_areas[qy * 2 + qx] += area
        aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
        extreme_aspect += int(aspect >= 8.0)
    layout_cx = weighted_cx / max(total_rect_area, 1e-6)
    layout_cy = weighted_cy / max(total_rect_area, 1e-6)
    pin_cx, pin_cy = _pin_centroid(case)
    return {
        "occupancy_ratio": total_rect_area / bbox_area,
        "frame_utilization": bbox_area / max(frame.area, 1e-6),
        "left_right_balance": abs(left_area - right_area) / max(total_rect_area, 1e-6),
        "top_bottom_balance": abs(top_area - bottom_area) / max(total_rect_area, 1e-6),
        "quadrant_entropy": _normalized_entropy(quadrant_areas),
        "largest_empty_rectangle_ratio": _largest_empty_cell_ratio(placements, bbox),
        "placed_centroid_vs_pin_centroid_distance": math.hypot(
            layout_cx - pin_cx,
            layout_cy - pin_cy,
        ),
        "terminal_heavy_mean_distance_to_pin": _terminal_heavy_mean_distance(case, placements),
        "extreme_aspect_count": int(extreme_aspect),
        "fallback_count": int(fallback_count),
        **{
            f"frame_{key}": value
            for key, value in frame_diagnostics(case, placements, frame).items()
            if key in {"num_frame_violations", "max_protrusion_distance"}
        },
    }


def pathology_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "occupancy_ratio",
        "frame_utilization",
        "left_right_balance",
        "top_bottom_balance",
        "quadrant_entropy",
        "largest_empty_rectangle_ratio",
        "placed_centroid_vs_pin_centroid_distance",
        "terminal_heavy_mean_distance_to_pin",
        "extreme_aspect_count",
    )
    return {
        f"{key}_delta": float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in keys
    }


def hpwl_regression_per_boundary_gain(row: dict[str, Any]) -> float:
    gain = max(float(row.get("boundary_delta", 0.0)), 1e-6)
    return max(float(row.get("hpwl_delta", 0.0)), 0.0) / gain


def is_suspicious_selection(
    row: dict[str, Any],
    *,
    hpwl_norm_threshold: float = 0.25,
    hpwl_raw_threshold: float = 3.0,
) -> bool:
    normalized = normalized_move_row(
        {
            **row,
            "move_type": row.get("selected_move_type"),
            "before_metrics": {
                "hpwl_proxy": float(row.get("hpwl_delta", 0.0))
                / max(float(row.get("hpwl_delta_norm", 0.0)), 1e-6)
                if row.get("hpwl_delta_norm")
                else 0.0,
                "bbox_area": 1.0,
            },
        }
    )
    hpwl_norm = abs(float(row.get("hpwl_delta_norm", normalized["hpwl_delta_norm"])))
    return bool(
        row.get("selected_move_type") == "simple_compaction"
        and float(row.get("boundary_delta", 0.0)) > 0.0
        and (
            float(row.get("hpwl_delta", 0.0)) > hpwl_raw_threshold
            or hpwl_norm > hpwl_norm_threshold
        )
    )


def label_case_pathology(
    selection: dict[str, Any],
    before_pathology: dict[str, Any],
    after_pathology: dict[str, Any],
    *,
    hpwl_delta_norm: float,
    bbox_delta_norm: float,
) -> list[str]:
    labels: list[str] = []
    move = str(selection.get("selected_move_type"))
    boundary_delta = float(selection.get("boundary_delta", 0.0))
    hpwl_delta = float(selection.get("hpwl_delta", 0.0))
    bbox_delta = float(selection.get("bbox_delta", 0.0))
    soft_delta = float(selection.get("soft_delta", 0.0))
    path_delta = pathology_delta(before_pathology, after_pathology)
    if move == "original":
        labels.append("good_original")
    if move == "simple_compaction" and boundary_delta > 0 and hpwl_delta_norm <= 0.25:
        labels.append("good_compaction")
    if move == "simple_compaction" and boundary_delta > 0 and hpwl_delta_norm > 0.25:
        labels.append("compaction_boundary_gain_too_expensive")
    if (
        path_delta["left_right_balance_delta"] > 0.10
        or path_delta["top_bottom_balance_delta"] > 0.10
    ):
        labels.append("spatial_imbalance_after_compaction")
    if hpwl_delta > 0 and hpwl_delta_norm > 0.10:
        labels.append("hpwl_regression")
    if bbox_delta > 0 and bbox_delta_norm > 0.02:
        labels.append("bbox_regression")
    if move == "boundary_edge_reassign" and boundary_delta >= 0 and abs(hpwl_delta_norm) < 0.10:
        labels.append("boundary_edge_reassign_low_risk")
    if move == "group_boundary_touch_template" and boundary_delta > 0:
        labels.append("group_template_helpful")
    if int(after_pathology.get("extreme_aspect_count", 0)) > int(
        before_pathology.get("extreme_aspect_count", 0)
    ):
        labels.append("shape_fragmentation")
    if int(after_pathology.get("fallback_count", 0)) > 0:
        labels.append("candidate_family_fallback_risk")
    if not labels and boundary_delta >= 0 and bbox_delta <= 0 and soft_delta <= 0:
        labels.append("good_original" if move == "original" else f"good_{move}")
    return labels or ["unclassified"]


def scale_coverage_report(
    profiles: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    pathology_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_by_case = {int(row["case_id"]): row for row in selections}
    pathology_by_case = {int(row["case_id"]): row for row in pathology_rows}
    buckets: dict[str, dict[str, Any]] = {}
    for bucket in ("small", "medium", "large", "xl"):
        case_ids = [int(row["case_id"]) for row in profiles if row.get("size_bucket") == bucket]
        selected = [selection_by_case[idx] for idx in case_ids if idx in selection_by_case]
        pathologies = [pathology_by_case[idx] for idx in case_ids if idx in pathology_by_case]
        boundary_values = [float(row.get("boundary_delta", 0.0)) for row in selected]
        buckets[bucket] = {
            "case_count": len(case_ids),
            "selected_move_counts": dict(
                Counter(str(row["selected_move_type"]) for row in selected)
            ),
            "mean_boundary_delta": _mean(boundary_values),
            "median_boundary_delta": _median(boundary_values),
            "mean_normalized_hpwl_delta": _mean(
                [float(row.get("hpwl_delta_norm", 0.0)) for row in pathologies]
            ),
            "median_normalized_hpwl_delta": _median(
                [float(row.get("hpwl_delta_norm", 0.0)) for row in pathologies]
            ),
            "mean_normalized_bbox_delta": _mean(
                [float(row.get("bbox_delta_norm", 0.0)) for row in pathologies]
            ),
            "median_normalized_bbox_delta": _median(
                [float(row.get("bbox_delta_norm", 0.0)) for row in pathologies]
            ),
            "pathology_label_counts": dict(
                Counter(label for row in pathologies for label in row.get("pathology_labels", []))
            ),
            "coverage_gap": len(case_ids) == 0,
        }
    return {
        "buckets": buckets,
        "coverage_gap_note": (
            "large/xl absent from current Step6M 40-case prefix"
            if buckets["large"]["coverage_gap"] or buckets["xl"]["coverage_gap"]
            else "all size buckets represented"
        ),
    }


def guard_calibration_candidates(pathology_rows: list[dict[str, Any]]) -> dict[str, Any]:
    suspicious = [row for row in pathology_rows if row.get("suspicious")]
    labels = Counter(label for row in pathology_rows for label in row.get("pathology_labels", []))
    return {
        "candidate_guards": [
            {
                "guard": "reject simple_compaction if normalized_hpwl_delta > 0.25",
                "would_flag_cases": [
                    int(row["case_id"])
                    for row in pathology_rows
                    if row.get("selected_move_type") == "simple_compaction"
                    and float(row.get("hpwl_delta_norm", 0.0)) > 0.25
                ],
            },
            {
                "guard": "reject if hpwl_regression_per_boundary_gain > 40",
                "would_flag_cases": [
                    int(row["case_id"])
                    for row in pathology_rows
                    if float(row.get("hpwl_regression_per_boundary_gain", 0.0)) > 40.0
                ],
            },
            {
                "guard": "reject if spatial balance worsens by > 0.10 on either axis",
                "would_flag_cases": [
                    int(row["case_id"])
                    for row in pathology_rows
                    if float(row["pathology_delta"].get("left_right_balance_delta", 0.0)) > 0.10
                    or float(row["pathology_delta"].get("top_bottom_balance_delta", 0.0)) > 0.10
                ],
            },
            {
                "guard": "reject if bbox_delta_norm > 0.02 unless soft_delta < 0",
                "would_flag_cases": [
                    int(row["case_id"])
                    for row in pathology_rows
                    if float(row.get("bbox_delta_norm", 0.0)) > 0.02
                    and float(row.get("soft_delta", 0.0)) >= 0.0
                ],
            },
        ],
        "suspicious_case_ids": [int(row["case_id"]) for row in suspicious],
        "pathology_label_counts": dict(labels),
        "note": "Candidates are diagnostic proposals only; Step6N does not apply them.",
    }


def profile_rows_from_cases(
    cases: list[FloorSetCase],
    placements_by_case: dict[int, dict[int, Placement]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if idx not in placements_by_case:
            continue
        rows.append(profile_case(case, placements_by_case[idx], suite="reported"))
    return rows


def _pin_centroid(case: FloorSetCase) -> tuple[float, float]:
    if case.pins_pos.numel() == 0:
        return 0.0, 0.0
    xs = [float(row[0]) for row in case.pins_pos.tolist()]
    ys = [float(row[1]) for row in case.pins_pos.tolist()]
    return sum(xs) / max(len(xs), 1), sum(ys) / max(len(ys), 1)


def _terminal_heavy_mean_distance(
    case: FloorSetCase,
    placements: dict[int, Placement],
) -> float:
    distances: list[float] = []
    nearest_pin_cache: dict[int, tuple[float, float]] = {}
    for pin_idx, block_id, weight in case.p2b_edges.tolist():
        block = int(block_id)
        if block not in placements or abs(float(weight)) <= 0:
            continue
        if int(pin_idx) >= len(case.pins_pos):
            continue
        pin_row = case.pins_pos[int(pin_idx)].tolist()
        nearest_pin_cache[block] = (float(pin_row[0]), float(pin_row[1]))
    for block, pin in nearest_pin_cache.items():
        x, y, w, h = placements[block]
        bx, by = x + w / 2.0, y + h / 2.0
        distances.append(math.hypot(bx - pin[0], by - pin[1]))
    return sum(distances) / max(len(distances), 1)


def _normalized_entropy(values: list[float]) -> float:
    total = sum(max(value, 0.0) for value in values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        p = max(value, 0.0) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy / math.log(max(len(values), 2))


def _largest_empty_cell_ratio(
    placements: dict[int, Placement],
    bbox: tuple[float, float, float, float],
) -> float:
    xs = sorted(
        {
            bbox[0],
            bbox[2],
            *(x for x, _y, _w, _h in placements.values()),
            *(x + w for x, _y, w, _h in placements.values()),
        }
    )
    ys = sorted(
        {
            bbox[1],
            bbox[3],
            *(y for _x, y, _w, _h in placements.values()),
            *(y + h for _x, y, _w, h in placements.values()),
        }
    )
    max_empty = 0.0
    for x0, x1 in zip(xs, xs[1:], strict=False):
        if x1 <= x0:
            continue
        for y0, y1 in zip(ys, ys[1:], strict=False):
            if y1 <= y0:
                continue
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            occupied = any(
                x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in placements.values()
            )
            if not occupied:
                max_empty = max(max_empty, (x1 - x0) * (y1 - y0))
    return max_empty / max(_area(bbox), 1e-6)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    mid = len(rows) // 2
    if len(rows) % 2:
        return rows[mid]
    return (rows[mid - 1] + rows[mid]) / 2.0
