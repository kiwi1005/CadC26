from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.research.boundary_failure_attribution import block_role_flags
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _bbox_boundary_satisfied_edges,
    _bbox_from_placements,
    _boundary_edges,
    final_bbox_boundary_metrics,
)


def _block_code(case: FloorSetCase, block_id: int, column: ConstraintColumns) -> int:
    return int(float(case.constraints[block_id, column].item()))


def _is_fixed_or_preplaced(case: FloorSetCase, block_id: int) -> bool:
    return bool(_block_code(case, block_id, ConstraintColumns.FIXED)) or bool(
        _block_code(case, block_id, ConstraintColumns.PREPLACED)
    )


def _overlaps(
    placements: dict[int, Placement],
    block_id: int,
    candidate: Placement,
    *,
    sep: float = 1e-4,
) -> bool:
    x, y, w, h = candidate
    for other, box in placements.items():
        if other == block_id:
            continue
        ox, oy, ow, oh = box
        if max(x, ox) < min(x + w, ox + ow) - sep and max(y, oy) < min(y + h, oy + oh) - sep:
            return True
    return False


def _align_to_edge(
    edge: str,
    old_box: Placement,
    width: float,
    height: float,
    bbox: tuple[float, float, float, float],
) -> Placement:
    x, y, old_w, old_h = old_box
    cx = x + old_w / 2.0
    cy = y + old_h / 2.0
    xmin, ymin, xmax, ymax = bbox
    if edge == "left":
        return xmin, cy - height / 2.0, width, height
    if edge == "right":
        return xmax - width, cy - height / 2.0, width, height
    if edge == "bottom":
        return cx - width / 2.0, ymin, width, height
    if edge == "top":
        return cx - width / 2.0, ymax - height, width, height
    return cx - width / 2.0, cy - height / 2.0, width, height


def _shape_variants(
    area: float,
    old_box: Placement,
    required_edges: tuple[str, ...],
    bbox: tuple[float, float, float, float],
) -> list[tuple[str, float, float]]:
    _x, _y, old_w, old_h = old_box
    old_ratio = max(old_w / max(old_h, 1e-6), 1e-6)
    ratios = {
        "current_shape": old_ratio,
        "square": 1.0,
        "narrower_taller": max(old_ratio * 0.5, math.exp(-2.0)),
        "wider_shorter": min(old_ratio * 2.0, math.exp(2.0)),
    }
    if any(edge in {"left", "right"} for edge in required_edges):
        edge_span = max((bbox[3] - bbox[1]) * 0.25, math.sqrt(area))
        ratios["edge_slot_fitting"] = max(area / max(edge_span * edge_span, 1e-6), math.exp(-2.0))
    elif any(edge in {"bottom", "top"} for edge in required_edges):
        edge_span = max((bbox[2] - bbox[0]) * 0.25, math.sqrt(area))
        ratios["edge_slot_fitting"] = min(
            max(edge_span * edge_span / max(area, 1e-6), 1e-6),
            math.exp(2.0),
        )
    rows: list[tuple[str, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for name, ratio in ratios.items():
        width = math.sqrt(area * ratio)
        height = math.sqrt(area / max(ratio, 1e-6))
        key = (round(width * 10000), round(height * 10000))
        if key in seen:
            continue
        seen.add(key)
        rows.append((name, width, height))
    return rows


def shape_group_intervention_probes(
    case: FloorSetCase,
    placements: dict[int, Placement],
    failure_rows: list[dict[str, Any]],
    edge_owner_rows: list[dict[str, Any]],
    *,
    frame: PuzzleFrame | None = None,
) -> dict[str, Any]:
    """Run targeted Step6L-E probes without changing runtime behavior."""

    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return {
            "shape_probe_records": [],
            "mib_probe_records": [],
            "group_probe_records": [],
            "best_shape_probe_layout": None,
        }

    target_ids: set[int] = {int(row["block_id"]) for row in failure_rows}
    for row in edge_owner_rows:
        if row.get("regular_or_nonboundary_stole_edge"):
            target_ids.update(int(value) for value in row.get("owner_block_ids", []))

    baseline = final_bbox_boundary_metrics(case, placements)
    baseline_rate = float(baseline["final_bbox_boundary_satisfaction_rate"])
    shape_records: list[dict[str, Any]] = []
    best_layout: dict[int, Placement] | None = None
    best_gain = 0.0
    for block_id in sorted(target_ids):
        if block_id not in placements:
            continue
        flags = block_role_flags(case, block_id)
        code = int(flags["boundary_code"])
        required_edges = _boundary_edges(code)
        if _is_fixed_or_preplaced(case, block_id):
            shape_records.append(
                {
                    "block_id": block_id,
                    "probe_type": "soft_shape",
                    "probe_variant": "fixed_or_preplaced_skip",
                    "hard_feasible": True,
                    "accepted": False,
                    "reject_reason": "fixed_or_preplaced_exact",
                    "role_flags": flags,
                }
            )
            continue
        area = float(case.area_targets[block_id].item())
        old_box = placements[block_id]
        for variant, width, height in _shape_variants(area, old_box, required_edges, bbox):
            edge_choices = required_edges or ("center_preserve",)
            for edge in edge_choices:
                candidate = _align_to_edge(edge, old_box, width, height, bbox)
                reject_reasons: list[str] = []
                if frame is not None and not frame.contains_box(candidate):
                    reject_reasons.append("outside_frame")
                if _overlaps(placements, block_id, candidate):
                    reject_reasons.append("overlap")
                updated = dict(placements)
                if not reject_reasons:
                    updated[block_id] = candidate
                metrics = final_bbox_boundary_metrics(case, updated)
                gain = float(metrics["final_bbox_boundary_satisfaction_rate"]) - baseline_rate
                sat_edges = 0
                total_edges = 0
                updated_bbox = _bbox_from_placements(updated.values())
                if updated_bbox is not None and code:
                    sat_edges, total_edges = _bbox_boundary_satisfied_edges(
                        code, updated[block_id], updated_bbox
                    )
                accepted = not reject_reasons and gain >= 0.0
                record = {
                    "block_id": block_id,
                    "probe_type": "soft_shape",
                    "probe_variant": variant,
                    "target_edge": edge,
                    "old_shape": {"w": old_box[2], "h": old_box[3]},
                    "new_shape": {"w": width, "h": height},
                    "hard_feasible": not reject_reasons,
                    "accepted": accepted,
                    "reject_reasons": reject_reasons,
                    "boundary_delta": gain,
                    "block_satisfied_edges_after": sat_edges,
                    "block_required_edges": total_edges,
                    "role_flags": flags,
                    "candidate_layout": {
                        str(idx): list(box) for idx, box in updated.items()
                    }
                    if accepted and gain > 0
                    else None,
                }
                shape_records.append(record)
                if accepted and gain > best_gain:
                    best_gain = gain
                    best_layout = updated

    return {
        "shape_probe_records": shape_records,
        "mib_probe_records": _mib_probe_records(case, placements, target_ids),
        "group_probe_records": _group_probe_records(case, placements, target_ids),
        "best_shape_probe_layout": {
            str(idx): list(box) for idx, box in best_layout.items()
        }
        if best_layout is not None
        else None,
        "summary": {
            "target_block_count": len(target_ids),
            "shape_probe_count": len(shape_records),
            "accepted_shape_probe_count": sum(1 for row in shape_records if row.get("accepted")),
            "best_shape_boundary_gain": best_gain,
        },
    }


def _mib_probe_records(
    case: FloorSetCase,
    placements: dict[int, Placement],
    target_ids: set[int],
) -> list[dict[str, Any]]:
    by_mib: dict[int, list[int]] = defaultdict(list)
    for idx in target_ids:
        mib_id = _block_code(case, idx, ConstraintColumns.MIB)
        if mib_id > 0:
            by_mib[mib_id].append(idx)
    records: list[dict[str, Any]] = []
    for mib_id, members in sorted(by_mib.items()):
        all_members = [
            idx
            for idx in range(case.block_count)
            if _block_code(case, idx, ConstraintColumns.MIB) == mib_id
        ]
        intervals = [
            (
                0.99 * float(case.area_targets[idx].item()),
                1.01 * float(case.area_targets[idx].item()),
            )
            for idx in all_members
        ]
        lo = max((row[0] for row in intervals), default=0.0)
        hi = min((row[1] for row in intervals), default=-1.0)
        compatible = lo <= hi
        master_shapes = [
            {"block_id": idx, "w": placements[idx][2], "h": placements[idx][3]}
            for idx in all_members
            if idx in placements
        ]
        records.append(
            {
                "mib_group_id": mib_id,
                "targeted_boundary_members": sorted(members),
                "affected_blocks": all_members,
                "compatible_exact_shape": compatible,
                "compatible_area_interval": [lo, hi] if compatible else None,
                "num_compatible_subgroups": 1 if compatible else len(all_members),
                "old_shape_master": master_shapes[0] if master_shapes else None,
                "new_shape_master_options": master_shapes[:5],
                "boundary_delta": 0.0,
                "bbox_delta": 0.0,
                "hpwl_delta": 0.0,
                "new_mib_violation_delta": 0 if compatible else 1,
                "hard_feasible": True,
                "probe_recommendation": "try_shape_master_search"
                if compatible
                else "split_into_compatible_subgroups_or_shared_aspect",
            }
        )
    return records


def _group_probe_records(
    case: FloorSetCase,
    placements: dict[int, Placement],
    target_ids: set[int],
) -> list[dict[str, Any]]:
    by_group: dict[int, list[int]] = defaultdict(list)
    for idx in target_ids:
        group_id = _block_code(case, idx, ConstraintColumns.CLUSTER)
        if group_id > 0:
            by_group[group_id].append(idx)
    records: list[dict[str, Any]] = []
    for group_id, targeted in sorted(by_group.items()):
        members = [
            idx
            for idx in range(case.block_count)
            if _block_code(case, idx, ConstraintColumns.CLUSTER) == group_id
            and idx in placements
        ]
        before_components = _shared_edge_components(members, placements)
        records.append(
            {
                "group_id": group_id,
                "targeted_boundary_members": sorted(targeted),
                "affected_blocks": members,
                "old_template": "current",
                "new_template_options": [
                    "horizontal_chain",
                    "vertical_chain",
                    "compact_2d",
                    "star",
                    "two_lobe_terminal_biased",
                    "boundary_touching_group_macro",
                ],
                "connected_components_before": before_components,
                "connected_components_after": before_components,
                "boundary_delta": 0.0,
                "bbox_delta": 0.0,
                "hpwl_delta": 0.0,
                "overlap_repair_needed": False,
                "hard_feasible": True,
                "probe_recommendation": "try_boundary_touching_group_macro"
                if before_components == 1
                else "repair_group_connectivity_before_boundary_macro",
            }
        )
    return records


def _shared_edge_components(members: list[int], placements: dict[int, Placement]) -> int:
    if not members:
        return 0
    parent = {idx: idx for idx in members}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pos, idx in enumerate(members):
        for other in members[pos + 1 :]:
            if _share_edge(placements[idx], placements[other]):
                union(idx, other)
    return len({find(idx) for idx in members})


def _share_edge(a: Placement, b: Placement, *, eps: float = 1e-4) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    vertical_touch = abs(ax + aw - bx) <= eps or abs(bx + bw - ax) <= eps
    y_overlap = max(ay, by) < min(ay + ah, by + bh) - eps
    horizontal_touch = abs(ay + ah - by) <= eps or abs(by + bh - ay) <= eps
    x_overlap = max(ax, bx) < min(ax + aw, bx + bw) - eps
    return bool((vertical_touch and y_overlap) or (horizontal_touch and x_overlap))
