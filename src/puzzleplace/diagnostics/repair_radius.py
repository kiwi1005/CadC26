from __future__ import annotations

import math
from typing import Any

from puzzleplace.data import FloorSetCase
from puzzleplace.diagnostics.aspect import aspect_stats
from puzzleplace.diagnostics.region_topology import moved_component_size, moved_region_count
from puzzleplace.eval.violation import summarize_violation_profile
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.research.move_library import (
    grouping_violation_count,
    layout_metrics,
    metric_deltas,
    mib_violation_count,
)
from puzzleplace.research.pathology import layout_pathology_metrics
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame


def changed_blocks(
    baseline: dict[int, Placement],
    alternative: dict[int, Placement],
    *,
    eps: float = 1e-6,
) -> set[int]:
    changed: set[int] = set()
    for idx, before in baseline.items():
        after = alternative.get(idx)
        if after is None:
            continue
        if any(abs(a - b) > eps for a, b in zip(before, after, strict=False)):
            changed.add(idx)
    return changed


def overlap_blocks(positions: dict[int, Placement]) -> set[int]:
    rows = list(positions.items())
    out: set[int] = set()
    for left_idx, (left_block, left_box) in enumerate(rows):
        for right_block, right_box in rows[left_idx + 1 :]:
            if intersection_area(left_box, right_box) > 1e-9:
                out.add(left_block)
                out.add(right_block)
    return out


def frame_violation_blocks(positions: dict[int, Placement], frame: PuzzleFrame) -> set[int]:
    return {idx for idx, box in positions.items() if not frame.contains_box(box)}


def hard_summary(case: FloorSetCase, placements: dict[int, Placement]) -> dict[str, Any]:
    ordered = [placements[idx] for idx in range(case.block_count)]
    hard = summarize_hard_legality(case, ordered)
    profile = summarize_violation_profile(case, placements)
    return {
        "hard_feasible": hard.is_feasible,
        "overlap_count": hard.overlap_violations,
        "area_violations": hard.area_violations,
        "dimension_violations": hard.dimension_violations,
        "total_overlap_area": profile.total_overlap_area,
    }


def repair_radius_metrics(
    case: FloorSetCase,
    *,
    baseline: dict[int, Placement],
    before_repair: dict[int, Placement],
    after_repair: dict[int, Placement],
    frame: PuzzleFrame,
    source_move_type: str,
    repair_mode: str,
    repair_seed: set[int],
    repair_region: set[int],
    repair_radius_exceeded: bool,
    runtime_estimate_ms: float,
    reject_reason: str | None,
) -> dict[str, Any]:
    before_hard = hard_summary(case, before_repair)
    after_hard = hard_summary(case, after_repair)
    before_metrics = layout_metrics(case, baseline, frame)
    after_metrics = layout_metrics(case, after_repair, frame)
    deltas = metric_deltas(before_metrics, after_metrics)
    before_aspect = aspect_stats(baseline)
    after_aspect = aspect_stats(after_repair)
    before_path = layout_pathology_metrics(case, baseline, frame)
    after_path = layout_pathology_metrics(case, after_repair, frame)
    moved = changed_blocks(baseline, after_repair)
    displacements = []
    for idx in moved:
        bx, by, bw, bh = baseline[idx]
        ax, ay, _aw, _ah = after_repair[idx]
        displacements.append(
            math.hypot(
                (ax + bw / 2.0) - (bx + bw / 2.0),
                (ay + bh / 2.0) - (by + bh / 2.0),
            )
        )
    affected_regions = moved_region_count(baseline, after_repair, frame)
    return {
        "case_id": case.case_id,
        "case_size_bucket": _size_bucket(case.block_count),
        "source_move_type": source_move_type,
        "repair_mode": repair_mode,
        "hard_feasible_before": before_hard["hard_feasible"],
        "hard_feasible_after": after_hard["hard_feasible"],
        "overlap_count_before": before_hard["overlap_count"],
        "overlap_count_after": after_hard["overlap_count"],
        "frame_protrusion_after": float(after_metrics["frame_protrusion"]),
        "MIB_group_violation_after": int(
            mib_violation_count(case, after_repair) + grouping_violation_count(case, after_repair)
        ),
        "moved_block_count": len(moved),
        "moved_block_fraction": len(moved) / max(case.block_count, 1),
        "max_displacement": max(displacements, default=0.0),
        "mean_displacement": sum(displacements) / max(len(displacements), 1),
        "displacement_chain_length": moved_component_size(case, moved),
        "affected_region_count": affected_regions,
        "affected_region_fraction": affected_regions / 16.0,
        "repair_seed_count": len(repair_seed),
        "repair_expanded_count": max(len(repair_region) - len(repair_seed), 0),
        "repair_radius_exceeded": repair_radius_exceeded,
        "hpwl_delta_norm": deltas["hpwl_delta"] / max(float(before_metrics["hpwl_proxy"]), 1e-9),
        "bbox_delta_norm": deltas["bbox_delta"] / max(float(before_metrics["bbox_area"]), 1e-9),
        "boundary_delta": deltas["boundary_delta"],
        "aspect_pathology_delta": float(after_aspect["extreme_aspect_area_fraction_gt_1_5"])
        - float(before_aspect["extreme_aspect_area_fraction_gt_1_5"]),
        "hole_fragmentation_delta": float(after_path["largest_empty_rectangle_ratio"])
        - float(before_path["largest_empty_rectangle_ratio"]),
        "runtime_estimate": runtime_estimate_ms,
        "reject_reason": reject_reason,
    }


def repair_failure_attribution(row: dict[str, Any], *, cap_fraction: float) -> dict[str, Any]:
    evidence = {
        "remaining_overlap_count": row["overlap_count_after"],
        "unresolved_violation_type": _unresolved_violation_type(row),
        "required_extra_blocks": max(
            int(row["moved_block_count"]) - int(row["repair_seed_count"]),
            0,
        ),
        "region_capacity_shortage": max(float(row["affected_region_fraction"]) - 0.5, 0.0),
        "macro_members_excluded": int(row["MIB_group_violation_after"]),
        "cap_that_stopped_expansion": cap_fraction if row["repair_radius_exceeded"] else None,
    }
    if row["repair_radius_exceeded"]:
        label = "repair_window_too_large"
    elif (
        int(row["MIB_group_violation_after"]) > 0
        and row["repair_mode"] != "macro_component_repair"
    ):
        label = "macro_component_missing"
    elif int(row["overlap_count_after"]) > 0 and float(row["moved_block_fraction"]) <= cap_fraction:
        label = "legalizer_algorithm_insufficient"
    elif float(row["affected_region_fraction"]) > 0.5:
        label = "repair_window_too_large"
    elif row["repair_mode"] in {"geometry_window_repair", "region_cell_repair"}:
        label = "repair_window_too_small"
    else:
        label = "move_itself_incompatible"
    return {
        "case_id": row["case_id"],
        "source_move_type": row["source_move_type"],
        "repair_mode": row["repair_mode"],
        "label": label,
        "evidence": evidence,
    }


def pareto_repair_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [
        row
        for row in rows
        if row["hard_feasible_after"]
        and not row["repair_radius_exceeded"]
        and float(row["frame_protrusion_after"]) <= 1e-6
    ]
    if not feasible:
        return {"front": [], "representatives": {}}
    front = [row for row in feasible if not any(_dominates(other, row) for other in feasible)]
    representatives = {
        "min_radius": min(
            front,
            key=lambda row: (row["moved_block_fraction"], row["affected_region_count"]),
        ),
        "best_hpwl": min(front, key=lambda row: row["hpwl_delta_norm"]),
        "best_boundary": max(front, key=lambda row: row["boundary_delta"]),
        "closest_to_ideal": min(front, key=_normalized_sum(front)),
    }
    return {
        "front": [_compact(row) for row in front],
        "representatives": {key: _compact(value) for key, value in representatives.items()},
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    le_keys = (
        "moved_block_fraction",
        "affected_region_count",
        "hpwl_delta_norm",
        "bbox_delta_norm",
    )
    better_or_equal = all(float(left[key]) <= float(right[key]) + 1e-12 for key in le_keys)
    boundary_ok = float(left["boundary_delta"]) >= float(right["boundary_delta"]) - 1e-12
    strictly = any(float(left[key]) < float(right[key]) - 1e-12 for key in le_keys) or (
        float(left["boundary_delta"]) > float(right["boundary_delta"]) + 1e-12
    )
    return better_or_equal and boundary_ok and strictly


def _normalized_sum(rows: list[dict[str, Any]]):
    keys = (
        "moved_block_fraction",
        "affected_region_count",
        "hpwl_delta_norm",
        "bbox_delta_norm",
    )
    ranges = {
        key: (
            min(float(row[key]) for row in rows),
            max(float(row[key]) for row in rows),
        )
        for key in keys
    }
    boundary_values = [float(row["boundary_delta"]) for row in rows]
    boundary_range = (min(boundary_values), max(boundary_values))

    def key_fn(row: dict[str, Any]) -> float:
        total = 0.0
        for key in keys:
            lo, hi = ranges[key]
            total += (float(row[key]) - lo) / max(hi - lo, 1e-12)
        blo, bhi = boundary_range
        total += (bhi - float(row["boundary_delta"])) / max(bhi - blo, 1e-12)
        return total

    return key_fn


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "source_move_type",
        "repair_mode",
        "hard_feasible_after",
        "moved_block_fraction",
        "affected_region_count",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "boundary_delta",
        "repair_radius_exceeded",
        "reject_reason",
    )
    return {key: row[key] for key in keys}


def intersection_area(left: Placement, right: Placement) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return max(min(lx + lw, rx + rw) - max(lx, rx), 0.0) * max(
        min(ly + lh, ry + rh) - max(ly, ry),
        0.0,
    )


def _unresolved_violation_type(row: dict[str, Any]) -> str:
    if int(row["overlap_count_after"]) > 0:
        return "overlap"
    if float(row["frame_protrusion_after"]) > 1e-6:
        return "frame_protrusion"
    if int(row["MIB_group_violation_after"]) > 0:
        return "MIB_group"
    if not bool(row["hard_feasible_after"]):
        return "hard_unknown"
    return "none"


def _size_bucket(block_count: int) -> str:
    if block_count <= 40:
        return "small"
    if block_count <= 70:
        return "medium"
    if block_count <= 100:
        return "large"
    return "xl"
