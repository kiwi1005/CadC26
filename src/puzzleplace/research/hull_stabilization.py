from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from puzzleplace.data import FloorSetCase
from puzzleplace.research.boundary_failure_attribution import (
    block_role_flags,
    final_bbox_edge_owner_audit,
)
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _area,
    _bbox_from_placements,
)


def _aspect(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(box[2] - box[0], 1e-6) / max(box[3] - box[1], 1e-6)


def _center(box: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if box is None:
        return 0.0, 0.0
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _edge_value(edge: str, box: tuple[float, float, float, float]) -> float:
    if edge == "left":
        return box[0]
    if edge == "right":
        return box[2]
    if edge == "bottom":
        return box[1]
    if edge == "top":
        return box[3]
    raise ValueError(f"unknown edge: {edge}")


def _required_boundary_edges(code: int) -> tuple[str, ...]:
    edges: list[str] = []
    if code & 1:
        edges.append("left")
    if code & 2:
        edges.append("right")
    if code & 8:
        edges.append("bottom")
    if code & 4:
        edges.append("top")
    return tuple(edges)


def _bucket_external_ratio(value: float) -> str:
    if value < 0.25:
        return "low"
    if value < 0.50:
        return "medium"
    return "high"


def hull_drift_metrics(
    case: FloorSetCase,
    placements: dict[int, Placement],
    predicted_hull: PuzzleFrame,
    *,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """Compare Step6J predicted compact hull against the final bbox owners.

    This is diagnostic only.  It intentionally reports final-bbox ownership
    drift instead of applying a hard hull-owner rule.
    """

    final_bbox = _bbox_from_placements(placements.values())
    predicted_box = (
        predicted_hull.xmin,
        predicted_hull.ymin,
        predicted_hull.xmax,
        predicted_hull.ymax,
    )
    owner_rows = final_bbox_edge_owner_audit(case, placements, eps=eps)
    owner_by_edge = {str(row["edge"]): row for row in owner_rows}
    if final_bbox is None:
        return {
            "predicted_hull": _frame_payload(predicted_hull),
            "final_bbox": None,
            "edge_drifts": {},
            "drift_owner_block_ids": {},
            "drift_summary": {"has_final_bbox": False},
        }
    pred_center = _center(predicted_box)
    final_center = _center(final_bbox)
    edge_drifts = {
        edge: float(_edge_value(edge, final_bbox) - _edge_value(edge, predicted_box))
        for edge in ("left", "right", "bottom", "top")
    }
    drift_owner_ids: dict[str, list[int]] = {}
    drift_owner_roles: dict[str, dict[str, Any]] = {}
    for edge in ("left", "right", "bottom", "top"):
        row = owner_by_edge.get(edge, {})
        owners = [int(block_id) for block_id in row.get("owner_block_ids", [])]
        drift_owner_ids[edge] = owners
        drift_owner_roles[edge] = {
            str(block_id): block_role_flags(case, block_id) for block_id in owners
        }
    drift_edges = [
        edge for edge, drift in edge_drifts.items() if abs(float(drift)) > eps
    ]
    regular_steal_edges = [
        edge
        for edge in ("left", "right", "bottom", "top")
        if owner_by_edge.get(edge, {}).get("regular_or_nonboundary_stole_edge")
    ]
    mib_or_group_owner_edges = [
        edge
        for edge in ("left", "right", "bottom", "top")
        if owner_by_edge.get(edge, {}).get("owner_is_mib")
        or owner_by_edge.get(edge, {}).get("owner_is_grouping")
    ]
    return {
        "predicted_hull": _frame_payload(predicted_hull),
        "final_bbox": {
            "xmin": final_bbox[0],
            "ymin": final_bbox[1],
            "xmax": final_bbox[2],
            "ymax": final_bbox[3],
            "area": _area(final_bbox),
            "aspect": _aspect(final_bbox),
        },
        "predicted_hull_area": predicted_hull.area,
        "final_bbox_area": _area(final_bbox),
        "predicted_hull_to_final_bbox_area_ratio": predicted_hull.area
        / max(_area(final_bbox), 1e-6),
        "final_bbox_to_predicted_hull_area_ratio": _area(final_bbox)
        / max(predicted_hull.area, 1e-6),
        "center_shift_x": float(final_center[0] - pred_center[0]),
        "center_shift_y": float(final_center[1] - pred_center[1]),
        "aspect_delta": float(_aspect(final_bbox) - _aspect(predicted_box)),
        "edge_drifts": edge_drifts,
        "edge_drift_left": edge_drifts["left"],
        "edge_drift_right": edge_drifts["right"],
        "edge_drift_bottom": edge_drifts["bottom"],
        "edge_drift_top": edge_drifts["top"],
        "drift_owner_block_ids": drift_owner_ids,
        "drift_owner_roles": drift_owner_roles,
        "drift_owner_is_boundary": {
            edge: any(bool(flags["is_boundary"]) for flags in roles.values())
            for edge, roles in drift_owner_roles.items()
        },
        "drift_owner_is_regular": {
            edge: any(bool(flags["is_regular"]) for flags in roles.values())
            for edge, roles in drift_owner_roles.items()
        },
        "drift_owner_is_mib": {
            edge: any(bool(flags["is_mib"]) for flags in roles.values())
            for edge, roles in drift_owner_roles.items()
        },
        "drift_owner_is_grouping": {
            edge: any(bool(flags["is_grouping"]) for flags in roles.values())
            for edge, roles in drift_owner_roles.items()
        },
        "drift_summary": {
            "has_final_bbox": True,
            "num_drift_edges": len(drift_edges),
            "drift_edges": drift_edges,
            "regular_or_nonboundary_steal_edges": regular_steal_edges,
            "mib_or_group_owner_edges": mib_or_group_owner_edges,
            "looks_like_estimator_error": bool(
                abs(_area(final_bbox) / max(predicted_hull.area, 1e-6) - 1.0) > 0.25
                or abs(_aspect(final_bbox) / max(_aspect(predicted_box), 1e-6) - 1.0)
                > 0.25
            ),
        },
    }


def attribution_cooccurrence(
    failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build Step6L-B overlap tables from Step6K failure labels."""

    failure_x_role: Counter[str] = Counter()
    failure_x_edge: Counter[str] = Counter()
    failure_x_boundary_type: Counter[str] = Counter()
    failure_x_external_ratio_bucket: Counter[str] = Counter()
    reason_x_role: Counter[str] = Counter()
    highlighted: Counter[str] = Counter()

    for row in failure_rows:
        labels = [str(row.get("failure_type", "unknown"))]
        labels.extend(str(reason) for reason in row.get("failure_reasons", []))
        labels = sorted(set(labels))
        flags = dict(row.get("role_flags", {}))
        roles = _role_labels(flags)
        edges = [str(edge) for edge in row.get("unsatisfied_edges", [])] or ["none"]
        boundary_type = str(row.get("required_boundary_type", "none"))
        ratio = float(flags.get("terminal_ratio", 0.0))
        bucket = _bucket_external_ratio(ratio)
        for label in labels:
            for role in roles:
                failure_x_role[f"{label}|{role}"] += 1
                if label in row.get("failure_reasons", []):
                    reason_x_role[f"{label}|{role}"] += 1
            for edge in edges:
                failure_x_edge[f"{label}|{edge}"] += 1
            failure_x_boundary_type[f"{label}|{boundary_type}"] += 1
            failure_x_external_ratio_bucket[f"{label}|{bucket}"] += 1
        if "on_predicted_hull_but_not_final_bbox" in labels and bool(
            flags.get("is_grouping")
        ):
            highlighted["on_predicted_hull_but_not_final_bbox∩grouping"] += 1
        if "on_predicted_hull_but_not_final_bbox" in labels and bool(flags.get("is_mib")):
            highlighted["on_predicted_hull_but_not_final_bbox∩MIB"] += 1
        if "edge_stolen_by_regular_or_nonboundary" in labels and (
            bool(flags.get("is_mib")) or bool(flags.get("is_grouping"))
        ):
            highlighted["edge_stolen_by_regular_or_nonboundary∩MIB/grouping"] += 1
        if "edge_segment_conflict" in labels and bool(flags.get("is_grouping")):
            highlighted["edge_segment_conflict∩grouping"] += 1
        if "role_conflict_grouping" in labels and bool(flags.get("is_terminal_heavy")):
            highlighted["role_conflict_grouping∩boundary+terminal-heavy"] += 1

    total = max(len(failure_rows), 1)
    return {
        "failure_type_x_role_flag": _counter_payload(failure_x_role),
        "failure_type_x_edge": _counter_payload(failure_x_edge),
        "failure_type_x_boundary_type": _counter_payload(failure_x_boundary_type),
        "failure_type_x_external_ratio_bucket": _counter_payload(
            failure_x_external_ratio_bucket
        ),
        "failure_reason_x_role_flag": _counter_payload(reason_x_role),
        "highlighted_intersections": _counter_payload(highlighted),
        "summary": {
            "failure_rows": len(failure_rows),
            "hull_drift_with_grouping_or_mib": int(
                highlighted["on_predicted_hull_but_not_final_bbox∩grouping"]
                + highlighted["on_predicted_hull_but_not_final_bbox∩MIB"]
            ),
            "hull_drift_with_grouping_or_mib_fraction": float(
                (
                    highlighted["on_predicted_hull_but_not_final_bbox∩grouping"]
                    + highlighted["on_predicted_hull_but_not_final_bbox∩MIB"]
                )
                / total
            ),
        },
    }


def hull_stealing_guard_audit(
    case: FloorSetCase,
    edge_owner_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    *,
    low_bbox_expansion_threshold: float = 1e-4,
    high_external_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Audit where a selective guard would have discouraged hull stealing.

    The guard is not applied here.  Rows record whether a non-boundary owner
    would be a candidate for later soft demotion and why exceptions remain.
    """

    coverage_by_edge: dict[str, int] = defaultdict(int)
    for row in failure_rows:
        count = int(row.get("candidate_count_for_required_edge", 0))
        for edge in row.get("unsatisfied_edges", []):
            coverage_by_edge[str(edge)] += count

    records: list[dict[str, Any]] = []
    for edge_row in edge_owner_rows:
        edge = str(edge_row["edge"])
        if not edge_row.get("regular_or_nonboundary_stole_edge"):
            continue
        owner_ids = [int(value) for value in edge_row.get("owner_block_ids", [])]
        expansion = dict(edge_row.get("owner_bbox_expansion_contribution", {}))
        for owner_id in owner_ids:
            flags = block_role_flags(case, owner_id)
            if bool(flags["is_boundary"]):
                continue
            owner_expansion = float(expansion.get(str(owner_id), 0.0))
            exceptions = []
            if bool(flags["is_fixed"]) or bool(flags["is_preplaced"]):
                exceptions.append("fixed_or_preplaced_unavoidable")
            if float(flags["terminal_ratio"]) >= high_external_ratio:
                exceptions.append("high_external_ratio")
            if owner_expansion <= low_bbox_expansion_threshold:
                exceptions.append("low_bbox_expansion")
            if coverage_by_edge[edge] <= 0:
                exceptions.append("no_feasible_boundary_owner")
            guard_applied = not exceptions
            records.append(
                {
                    "edge": edge,
                    "candidate_would_steal_hull_edge": True,
                    "stolen_edge_type": edge,
                    "owner_block_id": owner_id,
                    "owner_role_flags": flags,
                    "boundary_owner_available": coverage_by_edge[edge] > 0,
                    "bbox_expansion_if_selected": owner_expansion,
                    "hpwl_gain_if_selected": None,
                    "guard_applied": guard_applied,
                    "guard_exception_reasons": exceptions,
                    "selected_after_guard": not guard_applied,
                    "unsatisfied_required_block_ids": edge_row.get(
                        "unsatisfied_required_block_ids", []
                    ),
                }
            )
    return records


def select_alternative(
    alternatives: list[dict[str, Any]],
    *,
    bbox_tolerance: float = 0.05,
    hpwl_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Step6L-F lexicographic selection over already-evaluated alternatives."""

    original = next(row for row in alternatives if row["alternative_type"] == "original")
    original_bbox = float(original["bbox_area"])
    original_hpwl = float(original["hpwl_proxy"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in alternatives:
        reasons: list[str] = []
        if not row.get("hard_feasible", False):
            reasons.append("hard_infeasible")
        if float(row.get("frame_max_protrusion_distance", 0.0)) > 1e-4:
            reasons.append("frame_protrusion")
        if int(row.get("frame_num_violations", 0)) > 0:
            reasons.append("frame_violations")
        if float(row["bbox_area"]) > original_bbox * (1.0 + bbox_tolerance):
            reasons.append("bbox_worse_than_threshold")
        if float(row["hpwl_proxy"]) > original_hpwl * (1.0 + hpwl_tolerance):
            reasons.append("hpwl_catastrophically_worse")
        candidate = {**row, "reject_reasons": reasons}
        if reasons:
            rejected.append(candidate)
        else:
            accepted.append(candidate)

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            -float(row["boundary_satisfaction_rate"]),
            float(row["bbox_area"]),
            float(row["hpwl_proxy"]),
            float(row.get("soft_boundary_violations", 0.0)),
            float(row.get("role_conflicts_created", 0.0)),
        )

    selected = min(accepted, key=key) if accepted else original
    return {
        "selected_alternative_type": selected["alternative_type"],
        "selection_reason": _selection_reason(original, selected),
        "boundary_gain": float(
            selected["boundary_satisfaction_rate"] - original["boundary_satisfaction_rate"]
        ),
        "bbox_delta": float(selected["bbox_area"] - original_bbox),
        "hpwl_delta": float(selected["hpwl_proxy"] - original_hpwl),
        "soft_boundary_delta": float(
            selected.get("soft_boundary_violations", 0)
            - original.get("soft_boundary_violations", 0)
        ),
        "role_conflicts_resolved": int(selected.get("role_conflicts_resolved", 0)),
        "role_conflicts_created": int(selected.get("role_conflicts_created", 0)),
        "accepted_alternative_types": [row["alternative_type"] for row in accepted],
        "rejected_alternatives": rejected,
        "selected_metrics": selected,
    }


def _selection_reason(original: dict[str, Any], selected: dict[str, Any]) -> str:
    if selected["alternative_type"] == "original":
        return "original_kept_by_reject_or_no_gain"
    if selected["boundary_satisfaction_rate"] > original["boundary_satisfaction_rate"]:
        return "boundary_satisfaction_improved"
    if selected["bbox_area"] < original["bbox_area"]:
        return "bbox_improved_without_boundary_loss"
    return "lexicographic_tie_break"


def _role_labels(flags: dict[str, Any]) -> list[str]:
    labels = ["boundary" if flags.get("is_boundary") else "non_boundary"]
    if flags.get("is_grouping"):
        labels.append("grouping")
    if flags.get("is_mib"):
        labels.append("mib")
    if flags.get("is_terminal_heavy"):
        labels.append("terminal_heavy")
    if flags.get("is_fixed") or flags.get("is_preplaced"):
        labels.append("fixed_or_preplaced")
    if flags.get("is_regular"):
        labels.append("regular")
    if flags.get("multiple_roles"):
        labels.append("multiple_roles")
    return labels


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _frame_payload(frame: PuzzleFrame) -> dict[str, float | int | str]:
    return {
        "variant": frame.variant,
        "relaxation": frame.relaxation,
        "xmin": frame.xmin,
        "ymin": frame.ymin,
        "xmax": frame.xmax,
        "ymax": frame.ymax,
        "area": frame.area,
        "aspect": frame.width / max(frame.height, 1e-6),
    }
