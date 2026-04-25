from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.aspect import abs_log_aspect, block_area
from puzzleplace.repair import finalize_layout
from puzzleplace.research.boundary_failure_attribution import block_role_flags
from puzzleplace.research.move_library import layout_metrics, metric_deltas
from puzzleplace.research.pathology import layout_pathology_metrics
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

ShapePolicyName = Literal[
    "original_shape_policy",
    "mild_global_cap",
    "role_aware_cap",
    "filler_only_extreme",
    "boundary_strict_cap",
    "boundary_edge_slot_exception",
    "MIB_shape_master_regularized",
    "group_macro_aspect_regularized",
]

SHAPE_POLICIES: tuple[ShapePolicyName, ...] = (
    "original_shape_policy",
    "mild_global_cap",
    "role_aware_cap",
    "filler_only_extreme",
    "boundary_strict_cap",
    "boundary_edge_slot_exception",
    "MIB_shape_master_regularized",
    "group_macro_aspect_regularized",
)


@dataclass(frozen=True, slots=True)
class RoleShapeDecision:
    block_id: int
    log_aspect_cap: float | None
    role_trigger: str
    role_reason: str
    flags: dict[str, Any]


def external_ratio(case: FloorSetCase, block_id: int) -> float:
    external = sum(
        abs(float(weight))
        for pin_idx, dst, weight in case.p2b_edges.tolist()
        if int(dst) == block_id and int(pin_idx) >= 0
    )
    internal = sum(
        abs(float(weight))
        for src, dst, weight in case.b2b_edges.tolist()
        if int(src) == block_id or int(dst) == block_id
    )
    return external / max(external + internal, 1e-9)


def b2b_degree(case: FloorSetCase, block_id: int) -> float:
    return sum(
        abs(float(weight))
        for src, dst, weight in case.b2b_edges.tolist()
        if int(src) == block_id or int(dst) == block_id
    )


def deterministic_role_flags(
    case: FloorSetCase,
    block_id: int,
    placements: dict[int, Placement] | None = None,
) -> dict[str, Any]:
    flags = dict(block_role_flags(case, block_id))
    areas = [float(value) for value in case.area_targets.tolist()]
    degrees = [b2b_degree(case, idx) for idx in range(case.block_count)]
    area = float(case.area_targets[block_id].item())
    degree = b2b_degree(case, block_id)
    flags["external_ratio"] = external_ratio(case, block_id)
    flags["b2b_degree"] = degree
    flags["is_core"] = area >= percentile(areas, 0.75) or degree >= percentile(degrees, 0.75)
    flags["is_filler"] = (
        area <= percentile(areas, 0.25)
        and degree <= percentile(degrees, 0.50)
        and not flags["is_boundary"]
        and not flags["is_mib"]
        and not flags["is_grouping"]
        and not flags["is_fixed"]
        and not flags["is_preplaced"]
    )
    if placements and block_id in placements:
        flags["current_abs_log_aspect"] = abs_log_aspect(placements[block_id])
    return flags


def shape_policy_decision(
    policy: ShapePolicyName,
    case: FloorSetCase,
    block_id: int,
    placements: dict[int, Placement] | None = None,
) -> RoleShapeDecision:
    flags = deterministic_role_flags(case, block_id, placements)
    if flags["is_fixed"] or flags["is_preplaced"]:
        return RoleShapeDecision(
            block_id,
            None,
            "fixed_or_preplaced",
            "fixed/preplaced dimensions remain exact",
            flags,
        )
    if policy == "original_shape_policy":
        return RoleShapeDecision(block_id, None, "original", "no cap applied", flags)
    if policy == "mild_global_cap":
        return RoleShapeDecision(block_id, 2.0, "global", "mild global abs(log_r) cap", flags)
    if policy == "filler_only_extreme":
        cap = 3.0 if flags["is_filler"] else 2.0
        return RoleShapeDecision(
            block_id,
            cap,
            "filler" if flags["is_filler"] else "non_filler",
            "filler may remain extreme; others capped at regular range",
            flags,
        )
    if policy == "boundary_strict_cap":
        cap = 1.2 if flags["is_boundary"] else 2.0
        return RoleShapeDecision(
            block_id,
            cap,
            "boundary" if flags["is_boundary"] else "regular",
            "strict boundary shape cap without edge-slot exception",
            flags,
        )
    if policy == "boundary_edge_slot_exception":
        if flags["is_boundary"] and (flags["external_ratio"] >= 0.45 or flags["is_filler"]):
            return RoleShapeDecision(
                block_id,
                3.0,
                "boundary_edge_slot_exception",
                "boundary pin/slot-like block keeps wide aspect budget",
                flags,
            )
        if flags["is_boundary"]:
            return RoleShapeDecision(
                block_id, 1.2, "boundary", "boundary block capped unless exception applies", flags
            )
        return RoleShapeDecision(block_id, 2.0, "regular", "non-boundary regular cap", flags)
    if policy == "MIB_shape_master_regularized":
        cap = 1.5 if flags["is_mib"] else 2.0
        return RoleShapeDecision(
            block_id,
            cap,
            "MIB" if flags["is_mib"] else "regular",
            "MIB members use regularized shape-master cap",
            flags,
        )
    if policy == "group_macro_aspect_regularized":
        cap = 1.5 if flags["is_grouping"] else 2.0
        return RoleShapeDecision(
            block_id,
            cap,
            "grouping" if flags["is_grouping"] else "regular",
            "group members use template-level aspect cap proxy",
            flags,
        )
    # role_aware_cap
    if flags["is_mib"]:
        return RoleShapeDecision(
            block_id, 1.5, "MIB", "MIB shape master should stay regular when compatible", flags
        )
    if flags["is_grouping"]:
        return RoleShapeDecision(
            block_id, 1.5, "grouping", "group macro/member aspect regularization", flags
        )
    if flags["is_boundary"]:
        return RoleShapeDecision(
            block_id, 1.2, "boundary", "boundary block strict cap by default", flags
        )
    if flags["is_core"] or flags["is_terminal_heavy"]:
        return RoleShapeDecision(
            block_id,
            1.5,
            "core_or_terminal",
            "core/terminal-heavy blocks capped to reduce pathology",
            flags,
        )
    if flags["is_filler"]:
        return RoleShapeDecision(
            block_id, 3.0, "filler", "filler may use edge-slot-like extreme aspect", flags
        )
    return RoleShapeDecision(block_id, 2.0, "regular", "regular block aspect cap", flags)


def cap_for_block(
    policy: ShapePolicyName,
    case: FloorSetCase,
    block_id: int,
    placements: dict[int, Placement] | None = None,
) -> float | None:
    return shape_policy_decision(policy, case, block_id, placements).log_aspect_cap


def capped_shape(
    area: float, current_w: float, current_h: float, cap: float | None
) -> tuple[float, float]:
    if cap is None:
        return current_w, current_h
    signed = math.log(max(current_w, 1e-9) / max(current_h, 1e-9))
    clipped = max(min(signed, cap), -cap)
    ratio = math.exp(clipped)
    width = math.sqrt(max(area, 1e-9) * ratio)
    height = math.sqrt(max(area, 1e-9) / ratio)
    return width, height


def posthoc_shape_probe(
    policy: ShapePolicyName,
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> tuple[dict[int, Placement], list[dict[str, Any]]]:
    proposed: dict[int, Placement] = {}
    reasons: list[dict[str, Any]] = []
    for block_id, box in placements.items():
        x, y, w, h = box
        decision = shape_policy_decision(policy, case, block_id, placements)
        area = float(case.area_targets[block_id].item())
        new_w, new_h = capped_shape(area, w, h, decision.log_aspect_cap)
        cx, cy = x + w / 2.0, y + h / 2.0
        proposed[block_id] = (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)
        if abs(new_w - w) > 1e-6 or abs(new_h - h) > 1e-6:
            reasons.append(role_cap_reason(decision, box, proposed[block_id]))
    repaired = finalize_layout(case, proposed).positions
    repaired_map = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repaired)
    }
    return repaired_map, reasons


def role_cap_reason(
    decision: RoleShapeDecision, before: Placement, after: Placement
) -> dict[str, Any]:
    return {
        "block_id": decision.block_id,
        "original_w_h": [float(before[2]), float(before[3])],
        "new_w_h": [float(after[2]), float(after[3])],
        "original_abs_log_aspect": abs_log_aspect(before),
        "new_abs_log_aspect": abs_log_aspect(after),
        "role_trigger": decision.role_trigger,
        "role_reason": decision.role_reason,
        "boundary": bool(decision.flags["is_boundary"]),
        "MIB": bool(decision.flags["is_mib"]),
        "group": bool(decision.flags["is_grouping"]),
        "fixed": bool(decision.flags["is_fixed"]),
        "preplaced": bool(decision.flags["is_preplaced"]),
    }


def shape_policy_eval_row(
    *,
    case: FloorSetCase,
    policy: ShapePolicyName,
    track: str,
    baseline: dict[int, Placement],
    alternative: dict[int, Placement],
    frame: PuzzleFrame,
    role_cap_reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    before = layout_metrics(case, baseline, frame)
    after = layout_metrics(case, alternative, frame)
    deltas = metric_deltas(before, after)
    after_path = layout_pathology_metrics(case, alternative, frame)
    aspect_score = aspect_pathology_score(alternative)
    baseline_aspect_score = aspect_pathology_score(baseline)
    return {
        "case_id": parse_case_id(case.case_id),
        "track": track,
        "policy": policy,
        "before_metrics": before,
        "after_metrics": after,
        "boundary_violation_delta": -deltas["boundary_delta"],
        "boundary_delta": deltas["boundary_delta"],
        "hpwl_delta_norm": deltas["hpwl_delta"] / max(float(before["hpwl_proxy"]), 1e-9),
        "bbox_delta_norm": deltas["bbox_delta"] / max(float(before["bbox_area"]), 1e-9),
        "soft_delta": deltas["soft_delta"],
        "aspect_pathology_score": aspect_score,
        "aspect_pathology_delta": aspect_score - baseline_aspect_score,
        "hole_fragmentation": float(after_path["largest_empty_rectangle_ratio"]),
        "occupancy_ratio": float(after_path["occupancy_ratio"]),
        "disruption": disruption_cost(baseline, alternative),
        "hard_feasible": bool(after["hard_feasible"]),
        "frame_protrusion": float(after["frame_protrusion"]),
        "role_cap_count": len(role_cap_reasons),
    }


def aspect_pathology_score(placements: dict[int, Placement]) -> float:
    total_area = max(sum(block_area(box) for box in placements.values()), 1e-9)
    extreme_count = 0
    extreme_area = 0.0
    for box in placements.values():
        if abs_log_aspect(box) > 1.5:
            extreme_count += 1
            extreme_area += block_area(box)
    return extreme_count / max(len(placements), 1) + extreme_area / total_area


def disruption_cost(baseline: dict[int, Placement], alternative: dict[int, Placement]) -> float:
    moved = 0
    displacement = 0.0
    for block_id, box in baseline.items():
        if block_id not in alternative:
            continue
        ax, ay, aw, ah = alternative[block_id]
        bx, by, bw, bh = box
        before_center = (bx + bw / 2.0, by + bh / 2.0)
        after_center = (ax + aw / 2.0, ay + ah / 2.0)
        delta = math.hypot(after_center[0] - before_center[0], after_center[1] - before_center[1])
        displacement += delta / max(math.sqrt(block_area(box)), 1e-9)
        moved += int(delta > 1e-6 or abs(aw - bw) > 1e-6 or abs(ah - bh) > 1e-6)
    return moved / max(len(baseline), 1) + displacement / max(len(baseline), 1)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if row["hard_feasible"] and float(row.get("frame_protrusion", 0.0)) <= 1e-4
    ]
    front = []
    for row in valid:
        if not any(dominates(other, row) for other in valid if other is not row):
            front.append(row)
    return sorted(front, key=lambda row: (row["policy"], row["track"]))


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "boundary_violation_delta",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "aspect_pathology_score",
        "hole_fragmentation",
        "disruption",
    )
    return all(float(left[key]) <= float(right[key]) + 1e-12 for key in keys) and any(
        float(left[key]) < float(right[key]) - 1e-12 for key in keys
    )


def select_shape_policy_representatives(front: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not front:
        return {}
    return {
        "original": min(
            front, key=lambda row: (row["policy"] != "original_shape_policy", row["track"])
        ),
        "min_aspect_pathology": min(front, key=lambda row: row["aspect_pathology_score"]),
        "closest_to_ideal": min(front, key=normalized_objective_sum(front)),
        "min_disruption": min(front, key=lambda row: row["disruption"]),
        "best_hpwl": min(front, key=lambda row: row["hpwl_delta_norm"]),
        "best_boundary": min(front, key=lambda row: row["boundary_violation_delta"]),
    }


def normalized_objective_sum(rows: list[dict[str, Any]]):
    keys = (
        "boundary_violation_delta",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "aspect_pathology_score",
        "hole_fragmentation",
        "disruption",
    )
    ranges = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        ranges[key] = (min(values), max(values))

    def key_fn(row: dict[str, Any]) -> float:
        total = 0.0
        for key in keys:
            lo, hi = ranges[key]
            total += (float(row[key]) - lo) / max(hi - lo, 1e-12)
        return total

    return key_fn


def mib_group_policy_summary(case: FloorSetCase) -> dict[str, Any]:
    mib_groups: dict[int, list[int]] = {}
    for idx in range(case.block_count):
        mib_id = int(float(case.constraints[idx, ConstraintColumns.MIB].item()))
        if mib_id > 0:
            mib_groups.setdefault(mib_id, []).append(idx)
    exact_compatible = 0
    subgroup_count = 0
    invalid = 0
    for members in mib_groups.values():
        areas = [float(case.area_targets[idx].item()) for idx in members]
        if max(areas) <= min(areas) * 1.01:
            exact_compatible += 1
        else:
            subgroup_count += 1
    group_ids = {
        int(float(case.constraints[idx, ConstraintColumns.CLUSTER].item()))
        for idx in range(case.block_count)
        if int(float(case.constraints[idx, ConstraintColumns.CLUSTER].item())) > 0
    }
    return {
        "mib_exact_compatible_count": exact_compatible,
        "mib_subgroup_count": subgroup_count,
        "mib_alternative_invalid_count": invalid,
        "group_count": len(group_ids),
    }


def percentile(values: list[float], q: float) -> float:
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


def parse_case_id(case_id: int | str) -> int | str:
    text = str(case_id)
    if text.startswith("validation-"):
        return int(text.replace("validation-", ""))
    try:
        return int(text)
    except ValueError:
        return case_id
