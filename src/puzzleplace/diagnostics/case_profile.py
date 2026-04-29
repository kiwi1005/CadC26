from __future__ import annotations

from collections import Counter
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.aspect import aspect_stats, block_area
from puzzleplace.research.move_library import layout_metrics, size_bucket
from puzzleplace.research.pathology import layout_pathology_metrics
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame


def case_constraint_counts(case: FloorSetCase) -> dict[str, int]:
    mib_groups = {
        int(float(case.constraints[idx, ConstraintColumns.MIB].item()))
        for idx in range(case.block_count)
        if int(float(case.constraints[idx, ConstraintColumns.MIB].item())) > 0
    }
    group_ids = {
        int(float(case.constraints[idx, ConstraintColumns.CLUSTER].item()))
        for idx in range(case.block_count)
        if int(float(case.constraints[idx, ConstraintColumns.CLUSTER].item())) > 0
    }
    return {
        "terminal_count": len(case.pins_pos),
        "net_count": int(len(case.b2b_edges) + len(case.p2b_edges)),
        "fixed_preplaced_count": sum(
            int(
                bool(case.constraints[idx, ConstraintColumns.FIXED].item())
                or bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
            )
            for idx in range(case.block_count)
        ),
        "boundary_block_count": sum(
            int(bool(case.constraints[idx, ConstraintColumns.BOUNDARY].item()))
            for idx in range(case.block_count)
        ),
        "mib_count": sum(
            int(bool(case.constraints[idx, ConstraintColumns.MIB].item()))
            for idx in range(case.block_count)
        ),
        "mib_group_count": len(mib_groups),
        "grouping_count": sum(
            int(bool(case.constraints[idx, ConstraintColumns.CLUSTER].item()))
            for idx in range(case.block_count)
        ),
        "grouping_group_count": len(group_ids),
    }


def pin_bbox_area(case: FloorSetCase) -> float:
    if len(case.pins_pos) == 0:
        return 0.0
    xs = [float(row[0]) for row in case.pins_pos.tolist()]
    ys = [float(row[1]) for row in case.pins_pos.tolist()]
    return max(max(xs) - min(xs), 0.0) * max(max(ys) - min(ys), 0.0)


def candidate_family_summary(candidate_family_usage: dict[str, int]) -> dict[str, int]:
    usage = Counter(candidate_family_usage)
    return {
        "free_rect": _usage_for(usage, "free_rect"),
        "pin_pull": _usage_for(usage, "pin_pull"),
        "boundary": _usage_for(usage, "anchor:boundary"),
        "placed_block": _usage_for(usage, "anchor:placed_block"),
        "group_mate": _usage_for(usage, "group_mate"),
        "fallback": _usage_for(usage, "fallback"),
    }


def build_case_profile(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    candidate_family_usage: dict[str, int] | None = None,
    selected_step6p_representative: str | None = None,
    selected_step7b_representative: str | None = None,
) -> dict[str, Any]:
    counts = case_constraint_counts(case)
    layout = layout_metrics(case, placements, frame)
    pathology = layout_pathology_metrics(case, placements, frame)
    aspects = aspect_stats(placements)
    total_area = sum(block_area(box) for box in placements.values())
    family = candidate_family_summary(candidate_family_usage or {})
    profile: dict[str, Any] = {
        "case_id": parse_case_id(case.case_id),
        "block_count": case.block_count,
        "size_bucket": size_bucket(case.block_count),
        **counts,
        "area_utilization_proxy": total_area / max(float(frame.area), 1e-9),
        "pin_bbox_area": pin_bbox_area(case),
        "pin_bbox_to_virtual_frame_ratio": pin_bbox_area(case) / max(float(frame.area), 1e-9),
        "extreme_aspect_count": int(aspects["extreme_aspect_count_gt_1_5"]),
        "extreme_aspect_area_fraction": float(aspects["extreme_aspect_area_fraction_gt_1_5"]),
        "extreme_aspect_count_gt_2_0": int(aspects["extreme_aspect_count_gt_2_0"]),
        "hole_fragmentation_proxy": float(pathology["largest_empty_rectangle_ratio"]),
        "occupancy_ratio": float(pathology["occupancy_ratio"]),
        "boundary_failure_rate": 1.0 - float(layout["boundary_satisfaction_rate"]),
        "candidate_family_usage": candidate_family_usage or {},
        "candidate_family_summary": family,
        "selected_step6p_representative": selected_step6p_representative,
        "selected_step7b_representative": selected_step7b_representative,
    }
    profile["pathology_labels"] = pathology_labels(profile)
    return profile


def pathology_labels(profile: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if float(profile.get("extreme_aspect_area_fraction", 0.0)) >= 0.25 or int(
        profile.get("extreme_aspect_count", 0)
    ) >= max(8, int(profile.get("block_count", 0)) // 4):
        labels.append("aspect-heavy")
    if float(profile.get("boundary_failure_rate", 0.0)) >= 0.50:
        labels.append("boundary-heavy")
    if float(profile.get("mib_count", 0)) + float(profile.get("grouping_count", 0)) >= max(
        8, float(profile.get("block_count", 0)) * 0.25
    ):
        labels.append("MIB/group-heavy")
    if float(profile.get("area_utilization_proxy", 0.0)) <= 0.35:
        labels.append("sparse")
    if float(profile.get("hole_fragmentation_proxy", 0.0)) >= 0.35:
        labels.append("fragmented")
    return labels or ["unclassified"]


def profile_summary_by_bucket(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in ("small", "medium", "large", "xl"):
        rows = [row for row in profiles if row.get("size_bucket") == bucket]
        out[bucket] = {
            "case_count": len(rows),
            "mean_extreme_aspect_count": _mean(row["extreme_aspect_count"] for row in rows),
            "mean_extreme_aspect_area_fraction": _mean(
                row["extreme_aspect_area_fraction"] for row in rows
            ),
            "mean_boundary_failure_rate": _mean(row["boundary_failure_rate"] for row in rows),
            "mean_hole_fragmentation_proxy": _mean(row["hole_fragmentation_proxy"] for row in rows),
        }
    return out


def profile_summary_by_pathology(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    labels = sorted({label for row in profiles for label in row.get("pathology_labels", [])})
    for label in labels:
        rows = [row for row in profiles if label in row.get("pathology_labels", [])]
        out[label] = {
            "case_count": len(rows),
            "case_ids": [row["case_id"] for row in rows],
            "size_buckets": dict(Counter(str(row["size_bucket"]) for row in rows)),
            "mean_extreme_aspect_area_fraction": _mean(
                row["extreme_aspect_area_fraction"] for row in rows
            ),
            "mean_boundary_failure_rate": _mean(row["boundary_failure_rate"] for row in rows),
        }
    return out


def _usage_for(counter: Counter[str], needle: str) -> int:
    return sum(count for key, count in counter.items() if needle in key)


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / max(len(vals), 1)


def parse_case_id(case_id: int | str) -> int | str:
    text = str(case_id)
    if text.startswith("validation-"):
        return int(text.replace("validation-", ""))
    try:
        return int(text)
    except ValueError:
        return case_id
