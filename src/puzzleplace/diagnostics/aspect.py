from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from puzzleplace.data import FloorSetCase
from puzzleplace.research.boundary_failure_attribution import block_role_flags
from puzzleplace.research.virtual_frame import Placement, _bbox_from_placements

DEFAULT_ASPECT_THRESHOLDS = (1.5, 2.0, 3.0)
ROLE_BUCKETS = (
    "boundary",
    "MIB",
    "grouping",
    "fixed/preplaced",
    "terminal-heavy",
    "core",
    "regular",
    "filler",
)


def abs_log_aspect(box: Placement) -> float:
    _x, _y, w, h = box
    return abs(math.log(max(float(w), 1e-9) / max(float(h), 1e-9)))


def block_area(box: Placement) -> float:
    _x, _y, w, h = box
    return max(float(w), 0.0) * max(float(h), 0.0)


def aspect_stats(
    placements: dict[int, Placement],
    *,
    thresholds: Iterable[float] = DEFAULT_ASPECT_THRESHOLDS,
) -> dict[str, Any]:
    values = [abs_log_aspect(box) for _idx, box in sorted(placements.items())]
    areas = [block_area(box) for _idx, box in sorted(placements.items())]
    total_area = max(sum(areas), 1e-9)
    rows: dict[str, Any] = {
        "block_count": len(values),
        "median_abs_log_aspect": percentile(values, 0.50),
        "p90_abs_log_aspect": percentile(values, 0.90),
        "max_abs_log_aspect": max(values, default=0.0),
    }
    for threshold in thresholds:
        key = threshold_key(threshold)
        extreme = [idx for idx, value in enumerate(values) if value > threshold]
        rows[f"extreme_aspect_count_{key}"] = len(extreme)
        rows[f"extreme_aspect_area_fraction_{key}"] = (
            sum(areas[idx] for idx in extreme) / total_area
        )
    rows["extreme_aspect_count"] = rows["extreme_aspect_count_gt_2_0"]
    rows["extreme_aspect_area_fraction"] = rows["extreme_aspect_area_fraction_gt_2_0"]
    return rows


def block_role_buckets(
    case: FloorSetCase, block_id: int, placements: dict[int, Placement]
) -> list[str]:
    flags = block_role_flags(case, block_id)
    roles: list[str] = []
    if flags["is_boundary"]:
        roles.append("boundary")
    if flags["is_mib"]:
        roles.append("MIB")
    if flags["is_grouping"]:
        roles.append("grouping")
    if flags["is_fixed"] or flags["is_preplaced"]:
        roles.append("fixed/preplaced")
    if flags["is_terminal_heavy"]:
        roles.append("terminal-heavy")
    if is_core_block(case, block_id):
        roles.append("core")
    if is_filler_block(case, block_id, placements):
        roles.append("filler")
    if not roles:
        roles.append("regular")
    return roles


def aspect_by_role(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    threshold: float = 2.0,
) -> dict[str, Any]:
    buckets: dict[str, list[tuple[int, Placement]]] = defaultdict(list)
    for block_id, box in placements.items():
        for role in block_role_buckets(case, block_id, placements):
            buckets[role].append((block_id, box))
    out: dict[str, Any] = {}
    for role in ROLE_BUCKETS:
        rows = buckets.get(role, [])
        sub = {block_id: box for block_id, box in rows}
        stats = aspect_stats(sub)
        stats["extreme_block_ids"] = [
            block_id for block_id, box in rows if abs_log_aspect(box) > threshold
        ]
        out[role] = stats
    return out


def shape_change_summary(
    before: dict[int, Placement],
    after: dict[int, Placement],
    *,
    rel_tol: float = 1e-6,
) -> dict[str, Any]:
    changed: list[int] = []
    max_abs_log_aspect_delta = 0.0
    for block_id, after_box in after.items():
        if block_id not in before:
            continue
        before_box = before[block_id]
        width_delta = abs(float(after_box[2]) - float(before_box[2]))
        height_delta = abs(float(after_box[3]) - float(before_box[3]))
        scale = max(abs(float(before_box[2])), abs(float(before_box[3])), 1.0)
        if width_delta > rel_tol * scale or height_delta > rel_tol * scale:
            changed.append(block_id)
        max_abs_log_aspect_delta = max(
            max_abs_log_aspect_delta,
            abs(abs_log_aspect(after_box) - abs_log_aspect(before_box)),
        )
    return {
        "shape_changed_count": len(changed),
        "shape_changed_block_ids": changed,
        "max_abs_log_aspect_delta": max_abs_log_aspect_delta,
    }


def case_aspect_pathology(
    case: FloorSetCase,
    *,
    pre_move_placements: dict[int, Placement],
    post_move_placements: dict[int, Placement],
    selected_representative: str,
    selected_move_type: str,
    candidate_family_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    pre = aspect_stats(pre_move_placements)
    post = aspect_stats(post_move_placements)
    by_role = aspect_by_role(case, post_move_placements)
    shape_change = shape_change_summary(pre_move_placements, post_move_placements)
    return {
        "case_id": case.case_id,
        "n_blocks": case.block_count,
        "selected_representative": selected_representative,
        "selected_move_type": selected_move_type,
        "median_abs_log_aspect": post["median_abs_log_aspect"],
        "p90_abs_log_aspect": post["p90_abs_log_aspect"],
        "max_abs_log_aspect": post["max_abs_log_aspect"],
        "extreme_aspect_count": post["extreme_aspect_count"],
        "extreme_aspect_area_fraction": post["extreme_aspect_area_fraction"],
        "extreme_aspect_by_role": by_role,
        "extreme_aspect_by_move": {
            selected_move_type: {
                "case_count": 1,
                "extreme_aspect_count": post["extreme_aspect_count"],
                "extreme_aspect_area_fraction": post["extreme_aspect_area_fraction"],
            }
        },
        "extreme_aspect_by_candidate_family": candidate_family_usage or {},
        "pre_move_aspect_stats": pre,
        "post_move_aspect_stats": post,
        "shape_changed_by_move": {selected_move_type: shape_change},
    }


def summarize_by_role(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in ROLE_BUCKETS:
        counts = [
            float(row["extreme_aspect_by_role"][role]["extreme_aspect_count"]) for row in case_rows
        ]
        area_fractions = [
            float(row["extreme_aspect_by_role"][role]["extreme_aspect_area_fraction"])
            for row in case_rows
        ]
        out[role] = {
            "mean_extreme_aspect_count": mean(counts),
            "max_extreme_aspect_count": max(counts, default=0.0),
            "mean_extreme_aspect_area_fraction": mean(area_fractions),
        }
    return out


def summarize_candidate_families(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    weighted_extreme: Counter[str] = Counter()
    for row in case_rows:
        usage = Counter(row.get("extreme_aspect_by_candidate_family", {}))
        total.update(usage)
        for family, count in usage.items():
            weighted_extreme[family] += int(count) * int(row.get("extreme_aspect_count", 0))
    return {
        family: {
            "construction_usage_count": int(count),
            "usage_weighted_extreme_aspect_count": int(weighted_extreme[family]),
        }
        for family, count in sorted(total.items())
    }


def correlation_report(
    rows: list[dict[str, Any]], metrics: list[dict[str, Any]]
) -> dict[str, float | None]:
    by_case = {int(row["case_id"]): row for row in metrics}
    pairs = [(row, by_case.get(int(row["case_id"]), {})) for row in rows]
    return {
        "extreme_aspect_count_vs_hole_fragmentation": pearson(
            [row["extreme_aspect_count"] for row, _metric in pairs],
            [metric.get("largest_empty_rectangle_ratio", 0.0) for _row, metric in pairs],
        ),
        "extreme_aspect_count_vs_occupancy_ratio": pearson(
            [row["extreme_aspect_count"] for row, _metric in pairs],
            [metric.get("occupancy_ratio", 0.0) for _row, metric in pairs],
        ),
        "extreme_aspect_count_vs_bbox_delta_norm": pearson(
            [row["extreme_aspect_count"] for row, _metric in pairs],
            [metric.get("bbox_delta_norm", 0.0) for _row, metric in pairs],
        ),
        "extreme_aspect_count_vs_hpwl_delta_norm": pearson(
            [row["extreme_aspect_count"] for row, _metric in pairs],
            [metric.get("hpwl_delta_norm", 0.0) for _row, metric in pairs],
        ),
        "extreme_aspect_count_vs_boundary_failure": pearson(
            [row["extreme_aspect_count"] for row, _metric in pairs],
            [metric.get("boundary_failure_rate", 0.0) for _row, metric in pairs],
        ),
    }


def is_core_block(case: FloorSetCase, block_id: int) -> bool:
    degree = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        if int(src) == block_id or int(dst) == block_id:
            degree += abs(float(weight))
    degrees = []
    for idx in range(case.block_count):
        total = 0.0
        for src, dst, weight in case.b2b_edges.tolist():
            if int(src) == idx or int(dst) == idx:
                total += abs(float(weight))
        degrees.append(total)
    return degree >= percentile(degrees, 0.75) and degree > 0.0


def is_filler_block(case: FloorSetCase, block_id: int, placements: dict[int, Placement]) -> bool:
    area = block_area(placements[block_id])
    areas = [block_area(box) for box in placements.values()]
    flags = block_role_flags(case, block_id)
    return area <= percentile(areas, 0.25) and not any(
        [
            flags["is_boundary"],
            flags["is_mib"],
            flags["is_grouping"],
            flags["is_fixed"],
            flags["is_preplaced"],
        ]
    )


def bbox_occupancy(placements: dict[int, Placement]) -> float:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return 0.0
    xmin, ymin, xmax, ymax = bbox
    bbox_area = max((xmax - xmin) * (ymax - ymin), 1e-9)
    return sum(block_area(box) for box in placements.values()) / bbox_area


def threshold_key(threshold: float) -> str:
    return f"gt_{str(threshold).replace('.', '_')}"


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / max(len(vals), 1)


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x = [float(value) for value in xs]
    y = [float(value) for value in ys]
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=False)) / (denom_x * denom_y)
