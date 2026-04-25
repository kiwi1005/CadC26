from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from puzzleplace.research.pathology import normalized_move_row

LEGALITY_REJECTION_REASONS = {
    "hard_infeasible",
    "frame_protrusion",
    "fixed_or_preplaced_exact",
    "area_invalid",
    "fixed_preplaced_invalid",
    "overlap_invalid",
}
GLOBAL_MOVE_TYPES = {"simple_compaction", "edge_aware_compaction"}
SOFT_SHAPE_MOVE_TYPES = {"soft_aspect_flip", "soft_shape_stretch"}
MIB_MOVE_TYPES = {"mib_master_aspect_flip", "mib_master_edge_slot_shape"}
GROUP_MOVE_TYPES = {
    "group_template_rotate",
    "group_template_mirror",
    "group_boundary_touch_template",
    "cluster_split_or_two_lobe_repack",
}
BOUNDARY_MOVE_TYPES = {"boundary_edge_reassign"}
LOCAL_REGION_MOVE_TYPES = {"local_region_repack"}
COMBO_MOVE_TYPES = {
    "shape_then_light_compaction",
    "compaction_then_boundary_edge_reassign",
    "group_template_then_local_compaction",
}


def original_pareto_candidate(case_id: int | str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "move_type": "original",
        "target_blocks": [],
        "candidate_source": "original_inclusive_baseline",
        "boundary_delta": 0.0,
        "bbox_delta": 0.0,
        "hpwl_delta": 0.0,
        "soft_delta": 0.0,
        "grouping_delta": 0.0,
        "mib_delta": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
        "runtime_ms": 0.0,
        "before_metrics": {"hpwl_proxy": 1.0, "bbox_area": 1.0},
        "applicability_reason": "original_always_applicable",
    }


def target_roles(row: dict[str, Any]) -> list[dict[str, Any]]:
    roles = row.get("target_roles", {})
    if isinstance(roles, dict):
        return [dict(value) for value in roles.values() if isinstance(value, dict)]
    return []


def rejection_reason_set(row: dict[str, Any]) -> set[str]:
    reasons = row.get("rejected_reason", [])
    if isinstance(reasons, str):
        return {reasons}
    return {str(reason) for reason in reasons}


def is_hard_valid_alternative(row: dict[str, Any]) -> bool:
    if not bool(row.get("hard_feasible", False)):
        return False
    if float(row.get("frame_protrusion", 0.0)) > 1e-4:
        return False
    return not (rejection_reason_set(row) & LEGALITY_REJECTION_REASONS)


def applicability_filter_reason(row: dict[str, Any]) -> str:
    """Return keep/reject reason for Step6P's architecture-level move filter."""
    move_type = str(row.get("move_type"))
    reasons = rejection_reason_set(row)
    roles = target_roles(row)
    if "no_effect" in reasons:
        return "reject_no_effect_move"
    if move_type in GLOBAL_MOVE_TYPES:
        return "keep_global_alternative"
    if move_type in SOFT_SHAPE_MOVE_TYPES:
        if any(not role.get("is_fixed") and not role.get("is_preplaced") for role in roles):
            return "keep_nonfixed_soft_target"
        return "reject_soft_shape_not_nonfixed_soft"
    if move_type in MIB_MOVE_TYPES:
        if any(role.get("is_mib") for role in roles):
            return "keep_mib_compatible_target"
        return "reject_mib_move_without_mib_target"
    if move_type in GROUP_MOVE_TYPES:
        if any(role.get("is_grouping") for role in roles):
            return "keep_grouping_target"
        return "reject_group_move_without_grouping_target"
    if move_type in BOUNDARY_MOVE_TYPES:
        if any(role.get("is_boundary") for role in roles):
            return "keep_boundary_failure_target"
        return "reject_boundary_move_without_boundary_target"
    if move_type in LOCAL_REGION_MOVE_TYPES:
        if any(
            role.get("is_boundary")
            or role.get("is_mib")
            or role.get("is_grouping")
            or role.get("is_terminal_heavy")
            for role in roles
        ):
            return "keep_attribution_defined_local_region"
        return "reject_local_region_without_attribution_role"
    if move_type in COMBO_MOVE_TYPES:
        return "keep_depth2_combo"
    return "reject_unknown_move_type"


def is_applicable_alternative(row: dict[str, Any]) -> bool:
    return applicability_filter_reason(row).startswith("keep_")


def objective_values(row: dict[str, Any]) -> dict[str, float]:
    normalized = normalized_move_row(row)
    return {
        "boundary_violation_delta_norm": -float(row.get("boundary_delta", 0.0)),
        "hpwl_delta_norm": float(normalized["hpwl_delta_norm"]),
        "bbox_delta_norm": float(normalized["bbox_delta_norm"]),
        "disruption_cost_norm": disruption_cost_norm(row),
    }


def disruption_cost_norm(row: dict[str, Any]) -> float:
    if str(row.get("move_type")) == "original":
        return 0.0
    target_count = len(row.get("target_blocks", []) or [])
    profile = row.get("case_profile") or row.get("case_profile_tags") or {}
    block_count = float(profile.get("n_blocks", 0.0) or 0.0)
    moved_fraction = target_count / max(block_count, float(target_count), 1.0)
    pathology = row.get("pathology_delta", {})
    balance_worse = max(
        float(pathology.get("left_right_balance_delta", 0.0)),
        float(pathology.get("top_bottom_balance_delta", 0.0)),
        0.0,
    )
    combo_penalty = 0.05 if str(row.get("candidate_source")) == "estimated_depth2_combo" else 0.0
    return moved_fraction + balance_worse + combo_penalty


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_obj = left["objectives"]
    right_obj = right["objectives"]
    keys = (
        "boundary_violation_delta_norm",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "disruption_cost_norm",
    )
    return all(left_obj[key] <= right_obj[key] + 1e-12 for key in keys) and any(
        left_obj[key] < right_obj[key] - 1e-12 for key in keys
    )


def pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        if not any(dominates(other, candidate) for other in candidates if other is not candidate):
            front.append(candidate)
    return sorted(front, key=lambda row: _stable_candidate_key(row))


def build_pareto_candidates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"input_count": 0, "valid_count": 0, "applicable_count": 0}
    case_id = rows[0].get("case_id", "unknown")
    original = original_pareto_candidate(case_id)
    valid_rows = [row for row in rows if is_hard_valid_alternative(row)]
    no_effect_before = sum(int("no_effect" in rejection_reason_set(row)) for row in valid_rows)
    applicable_rows: list[dict[str, Any]] = []
    filter_counts: Counter[str] = Counter()
    for row in valid_rows:
        reason = applicability_filter_reason(row)
        filter_counts[reason] += 1
        if reason.startswith("keep_"):
            applicable_rows.append({**row, "applicability_reason": reason})
    combo_rows = synthesize_depth2_combos(applicable_rows)
    candidates = [original, *applicable_rows, *combo_rows]
    annotated = [{**row, "objectives": objective_values(row)} for row in candidates]
    no_effect_after = sum(int("no_effect" in rejection_reason_set(row)) for row in applicable_rows)
    return annotated, {
        "input_count": len(rows),
        "valid_count": len(valid_rows),
        "applicable_count": len(applicable_rows),
        "combo_count": len(combo_rows),
        "no_effect_before_filter": no_effect_before,
        "no_effect_after_filter": no_effect_after,
        "filter_reason_counts": dict(sorted(filter_counts.items())),
    }


def synthesize_depth2_combos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_move: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_move[str(row.get("move_type"))].append(row)
    combos: list[dict[str, Any]] = []
    combos.extend(
        _combine_pair(shape, compaction, "shape_then_light_compaction")
        for shape in [*by_move.get("soft_aspect_flip", []), *by_move.get("soft_shape_stretch", [])]
        for compaction in by_move.get("edge_aware_compaction", [])[:1]
    )
    combos.extend(
        _combine_pair(compaction, reassign, "compaction_then_boundary_edge_reassign")
        for compaction in [
            *by_move.get("simple_compaction", []),
            *by_move.get("edge_aware_compaction", []),
        ]
        for reassign in by_move.get("boundary_edge_reassign", [])
    )
    combos.extend(
        _combine_pair(group, compaction, "group_template_then_local_compaction")
        for group in [
            *by_move.get("group_template_rotate", []),
            *by_move.get("group_template_mirror", []),
            *by_move.get("group_boundary_touch_template", []),
        ]
        for compaction in by_move.get("edge_aware_compaction", [])[:1]
    )
    return combos[:24]


def _combine_pair(first: dict[str, Any], second: dict[str, Any], move_type: str) -> dict[str, Any]:
    target_blocks = list(
        dict.fromkeys([*first.get("target_blocks", []), *second.get("target_blocks", [])])
    )
    before = dict(first.get("before_metrics", {}))
    return {
        "case_id": first.get("case_id"),
        "suite": first.get("suite"),
        "move_type": move_type,
        "target_blocks": target_blocks,
        "target_roles": {
            **dict(first.get("target_roles", {})),
            **dict(second.get("target_roles", {})),
        },
        "candidate_source": "estimated_depth2_combo",
        "combo_parts": [first.get("move_type"), second.get("move_type")],
        "combo_part_targets": [
            list(first.get("target_blocks", [])),
            list(second.get("target_blocks", [])),
        ],
        "boundary_delta": float(first.get("boundary_delta", 0.0))
        + float(second.get("boundary_delta", 0.0)),
        "bbox_delta": float(first.get("bbox_delta", 0.0)) + float(second.get("bbox_delta", 0.0)),
        "hpwl_delta": float(first.get("hpwl_delta", 0.0)) + float(second.get("hpwl_delta", 0.0)),
        "soft_delta": float(first.get("soft_delta", 0.0)) + float(second.get("soft_delta", 0.0)),
        "grouping_delta": float(first.get("grouping_delta", 0.0))
        + float(second.get("grouping_delta", 0.0)),
        "mib_delta": float(first.get("mib_delta", 0.0)) + float(second.get("mib_delta", 0.0)),
        "hard_feasible": bool(
            first.get("hard_feasible", True) and second.get("hard_feasible", True)
        ),
        "frame_protrusion": max(
            float(first.get("frame_protrusion", 0.0)), float(second.get("frame_protrusion", 0.0))
        ),
        "runtime_ms": _runtime_ms(first) + _runtime_ms(second),
        "before_metrics": before,
        "case_profile": first.get("case_profile") or second.get("case_profile") or {},
        "applicability_reason": "keep_depth2_combo",
    }


def select_representatives(front: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not front:
        return {}
    ranges = objective_ranges(front)
    return {
        "min_disruption": min(
            front,
            key=lambda row: (
                row["objectives"]["disruption_cost_norm"],
                _objective_sum(row, ranges),
                _stable_candidate_key(row),
            ),
        ),
        "closest_to_ideal": min(
            front,
            key=lambda row: (
                _objective_sum(row, ranges),
                row["objectives"]["disruption_cost_norm"],
                _stable_candidate_key(row),
            ),
        ),
        "best_boundary": min(
            front,
            key=lambda row: (
                row["objectives"]["boundary_violation_delta_norm"],
                _stable_candidate_key(row),
            ),
        ),
        "best_hpwl": min(
            front,
            key=lambda row: (row["objectives"]["hpwl_delta_norm"], _stable_candidate_key(row)),
        ),
    }


def objective_ranges(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not candidates:
        return {}
    keys = candidates[0]["objectives"].keys()
    ranges: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row["objectives"][key]) for row in candidates]
        ranges[key] = {"min": min(values), "max": max(values), "span": max(values) - min(values)}
    return ranges


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "move_type",
        "target_blocks",
        "candidate_source",
        "combo_parts",
        "combo_part_targets",
        "boundary_delta",
        "hpwl_delta",
        "bbox_delta",
        "soft_delta",
        "grouping_delta",
        "mib_delta",
        "hard_feasible",
        "frame_protrusion",
        "runtime_ms",
        "applicability_reason",
        "objectives",
    )
    return {key: row[key] for key in keys if key in row}


def representative_to_selection(row: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "selected_move_type": row.get("move_type"),
        "selected_target_blocks": row.get("target_blocks", []),
        "selection_reason": f"step6p_{label}",
        "boundary_delta": float(row.get("boundary_delta", 0.0)),
        "bbox_delta": float(row.get("bbox_delta", 0.0)),
        "hpwl_delta": float(row.get("hpwl_delta", 0.0)),
        "soft_delta": float(row.get("soft_delta", 0.0)),
        "grouping_delta": float(row.get("grouping_delta", 0.0)),
        "mib_delta": float(row.get("mib_delta", 0.0)),
        "hard_feasible": bool(row.get("hard_feasible", True)),
        "frame_protrusion": float(row.get("frame_protrusion", 0.0)),
        "runtime_ms": _runtime_ms(row),
        "objectives": row.get("objectives", {}),
    }


def summarize_case_fronts(case_fronts: list[dict[str, Any]]) -> dict[str, Any]:
    closest = [
        row["representatives"]["closest_to_ideal"]
        for row in case_fronts
        if row.get("representatives")
    ]
    filter_rows = [row["filter_stats"] for row in case_fronts]
    return {
        "case_count": len(case_fronts),
        "closest_to_ideal_move_counts": dict(Counter(str(row["move_type"]) for row in closest)),
        "original_closest_to_ideal_count": sum(
            int(row["move_type"] == "original") for row in closest
        ),
        "simple_compaction_closest_to_ideal_count": sum(
            int(row["move_type"] == "simple_compaction") for row in closest
        ),
        "boundary_edge_reassign_available_case_count": sum(
            int(
                any(
                    candidate["move_type"] == "boundary_edge_reassign"
                    for candidate in row.get("pareto_front", [])
                )
            )
            for row in case_fronts
        ),
        "mean_front_size": _mean(row.get("front_size", 0) for row in case_fronts),
        "no_effect_before_filter": sum(
            int(row.get("no_effect_before_filter", 0)) for row in filter_rows
        ),
        "no_effect_after_filter": sum(
            int(row.get("no_effect_after_filter", 0)) for row in filter_rows
        ),
        "filter_reason_counts": dict(
            sum_counters(Counter(row.get("filter_reason_counts", {})) for row in filter_rows)
        ),
    }


def sum_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return total


def _runtime_ms(row: dict[str, Any]) -> float:
    if "runtime_ms" in row:
        return float(row.get("runtime_ms", 0.0))
    return (
        float(row.get("generation_time_ms", 0.0))
        + float(row.get("repair_time_ms", 0.0))
        + float(row.get("eval_time_ms", 0.0))
    )


def _stable_candidate_key(
    row: dict[str, Any],
) -> tuple[str, tuple[int, ...], float, float, float, float]:
    return (
        str(row.get("move_type")),
        tuple(int(value) for value in row.get("target_blocks", []) or []),
        float(row.get("boundary_delta", 0.0)),
        float(row.get("hpwl_delta", 0.0)),
        float(row.get("bbox_delta", 0.0)),
        _runtime_ms(row),
    )


def _objective_sum(row: dict[str, Any], ranges: dict[str, dict[str, float]]) -> float:
    total = 0.0
    for key, value in row["objectives"].items():
        span = max(float(ranges[key]["span"]), 1e-12)
        total += (float(value) - float(ranges[key]["min"])) / span
    return total


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    return sum(values_list) / max(len(values_list), 1)
