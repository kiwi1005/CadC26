from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _area,
    _bbox_boundary_satisfied_edges,
    _bbox_from_placements,
    _boundary_edges,
    boundary_frame_satisfied_edges,
    final_bbox_boundary_metrics,
)

_EDGE_TO_CODE = {"left": 1, "right": 2, "top": 4, "bottom": 8}


@dataclass(frozen=True, slots=True)
class CandidateEdgeCoverage:
    by_block: dict[int, dict[str, int]]

    def required_count(self, block_id: int, edge: str) -> int:
        return int(self.by_block.get(block_id, {}).get(edge, 0))

    def total_required_count(self, block_id: int, edges: list[str]) -> int:
        return sum(self.required_count(block_id, edge) for edge in edges)


def _block_code(case: FloorSetCase, block_id: int, column: ConstraintColumns) -> int:
    return int(float(case.constraints[block_id, column].item()))


def _terminal_ratio(case: FloorSetCase, block_id: int) -> float:
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
    return external / max(external + internal, 1e-6)


def block_role_flags(case: FloorSetCase, block_id: int) -> dict[str, int | bool | float]:
    boundary_code = _block_code(case, block_id, ConstraintColumns.BOUNDARY)
    mib_id = _block_code(case, block_id, ConstraintColumns.MIB)
    cluster_id = _block_code(case, block_id, ConstraintColumns.CLUSTER)
    fixed = bool(_block_code(case, block_id, ConstraintColumns.FIXED))
    preplaced = bool(_block_code(case, block_id, ConstraintColumns.PREPLACED))
    terminal_ratio = _terminal_ratio(case, block_id)
    terminal_heavy = terminal_ratio >= 0.5
    non_boundary_roles = [
        bool(mib_id),
        bool(cluster_id),
        fixed,
        preplaced,
        terminal_heavy,
    ]
    role_count = int(bool(boundary_code)) + sum(int(role) for role in non_boundary_roles)
    return {
        "boundary_code": boundary_code,
        "is_boundary": bool(boundary_code),
        "is_mib": bool(mib_id),
        "mib_id": mib_id,
        "is_grouping": bool(cluster_id),
        "cluster_id": cluster_id,
        "is_fixed": fixed,
        "is_preplaced": preplaced,
        "terminal_ratio": float(terminal_ratio),
        "is_terminal_heavy": bool(terminal_heavy),
        "is_regular": not any(
            [bool(boundary_code), bool(mib_id), bool(cluster_id), fixed, preplaced]
        ),
        "multiple_roles": role_count >= 2,
        "role_count": role_count,
    }


def _block_type(flags: dict[str, int | bool | float]) -> str:
    labels: list[str] = []
    if flags["is_boundary"]:
        labels.append("boundary")
    if flags["is_grouping"]:
        labels.append("grouping")
    if flags["is_mib"]:
        labels.append("mib")
    if flags["is_fixed"]:
        labels.append("fixed")
    if flags["is_preplaced"]:
        labels.append("preplaced")
    if flags["is_terminal_heavy"]:
        labels.append("terminal-heavy")
    return "+".join(labels) if labels else "regular"


def _edge_owned(
    edge: str, box: Placement, bbox: tuple[float, float, float, float], eps: float
) -> bool:
    x, y, w, h = box
    xmin, ymin, xmax, ymax = bbox
    if edge == "left":
        return abs(x - xmin) <= eps
    if edge == "right":
        return abs((x + w) - xmax) <= eps
    if edge == "bottom":
        return abs(y - ymin) <= eps
    if edge == "top":
        return abs((y + h) - ymax) <= eps
    raise ValueError(f"unknown edge: {edge}")


def _edge_interval(edge: str, box: Placement) -> tuple[float, float]:
    x, y, w, h = box
    if edge in {"left", "right"}:
        return y, y + h
    return x, x + w


def _leave_one_out_bbox_expansion(
    placements: dict[int, Placement], block_id: int, all_area: float
) -> float:
    without = {idx: box for idx, box in placements.items() if idx != block_id}
    return float(all_area - _area(_bbox_from_placements(without.values())))


def final_bbox_edge_owner_audit(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    eps: float = 1e-4,
) -> list[dict[str, Any]]:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return []
    all_area = _area(bbox)
    rows: list[dict[str, Any]] = []
    for edge in ("left", "right", "bottom", "top"):
        owner_ids = [
            int(idx) for idx, box in sorted(placements.items()) if _edge_owned(edge, box, bbox, eps)
        ]
        owner_flags = {idx: block_role_flags(case, idx) for idx in owner_ids}
        required_ids = [
            int(idx)
            for idx in range(case.block_count)
            if edge in _boundary_edges(_block_code(case, idx, ConstraintColumns.BOUNDARY))
        ]
        unsatisfied_required_ids = [
            idx
            for idx in required_ids
            if idx in placements and not _edge_owned(edge, placements[idx], bbox, eps)
        ]
        expansion = {
            str(idx): _leave_one_out_bbox_expansion(placements, idx, all_area) for idx in owner_ids
        }
        external = {str(idx): float(owner_flags[idx]["terminal_ratio"]) for idx in owner_ids}
        owner_required = [
            idx
            for idx in owner_ids
            if edge in _boundary_edges(_block_code(case, idx, ConstraintColumns.BOUNDARY))
        ]
        rows.append(
            {
                "edge": edge,
                "owner_block_ids": owner_ids,
                "owner_block_types": {str(idx): _block_type(owner_flags[idx]) for idx in owner_ids},
                "owner_is_boundary_required_for_edge": bool(owner_required),
                "owner_boundary_required_block_ids": owner_required,
                "owner_is_regular": any(bool(owner_flags[idx]["is_regular"]) for idx in owner_ids),
                "owner_is_mib": any(bool(owner_flags[idx]["is_mib"]) for idx in owner_ids),
                "owner_is_grouping": any(
                    bool(owner_flags[idx]["is_grouping"]) for idx in owner_ids
                ),
                "owner_external_ratio": external,
                "owner_external_ratio_mean": sum(external.values()) / max(len(external), 1),
                "owner_bbox_expansion_contribution": expansion,
                "owner_bbox_expansion_contribution_sum": sum(expansion.values()),
                "boundary_required_block_ids": required_ids,
                "unsatisfied_required_block_ids": unsatisfied_required_ids,
                "regular_or_nonboundary_stole_edge": bool(
                    unsatisfied_required_ids
                    and owner_ids
                    and any(
                        bool(owner_flags[idx]["is_regular"])
                        or not bool(owner_flags[idx]["is_boundary"])
                        for idx in owner_ids
                    )
                ),
            }
        )
    return rows


def boundary_role_overlap_audit(
    case: FloorSetCase,
    placements: dict[int, Placement],
) -> dict[str, Any]:
    bbox = _bbox_from_placements(placements.values())
    counts: Counter[str] = Counter()
    unsatisfied: Counter[str] = Counter()
    unsatisfied_blocks: list[int] = []
    for idx in range(case.block_count):
        flags = block_role_flags(case, idx)
        if not flags["is_boundary"]:
            continue
        keys = ["boundary_total_count"]
        extra_roles = 0
        if flags["is_grouping"]:
            keys.append("boundary_plus_grouping_count")
            extra_roles += 1
        if flags["is_mib"]:
            keys.append("boundary_plus_mib_count")
            extra_roles += 1
        if flags["is_terminal_heavy"]:
            keys.append("boundary_plus_terminal_heavy_count")
            extra_roles += 1
        if flags["is_fixed"] or flags["is_preplaced"]:
            keys.append("boundary_plus_fixed_or_preplaced_count")
            extra_roles += 1
        if extra_roles == 0:
            keys.append("boundary_only_count")
        if extra_roles >= 1:
            keys.append("boundary_plus_multiple_roles_count")
        for key in keys:
            counts[key] += 1
        satisfied = False
        if bbox is not None and idx in placements:
            sat, total = _bbox_boundary_satisfied_edges(
                int(flags["boundary_code"]), placements[idx], bbox
            )
            satisfied = total > 0 and sat == total
        if not satisfied:
            unsatisfied_blocks.append(idx)
            for key in keys:
                unsatisfied[key.replace("_count", "")] += 1
    return {
        "boundary_only_count": int(counts["boundary_only_count"]),
        "boundary_plus_grouping_count": int(counts["boundary_plus_grouping_count"]),
        "boundary_plus_mib_count": int(counts["boundary_plus_mib_count"]),
        "boundary_plus_terminal_heavy_count": int(counts["boundary_plus_terminal_heavy_count"]),
        "boundary_plus_fixed_or_preplaced_count": int(
            counts["boundary_plus_fixed_or_preplaced_count"]
        ),
        "boundary_plus_multiple_roles_count": int(counts["boundary_plus_multiple_roles_count"]),
        "boundary_total_count": int(counts["boundary_total_count"]),
        "unsatisfied_boundary_only": int(unsatisfied["boundary_only"]),
        "unsatisfied_boundary_plus_grouping": int(unsatisfied["boundary_plus_grouping"]),
        "unsatisfied_boundary_plus_mib": int(unsatisfied["boundary_plus_mib"]),
        "unsatisfied_boundary_plus_terminal_heavy": int(
            unsatisfied["boundary_plus_terminal_heavy"]
        ),
        "unsatisfied_boundary_plus_fixed_or_preplaced": int(
            unsatisfied["boundary_plus_fixed_or_preplaced"]
        ),
        "unsatisfied_boundary_plus_multiple_roles": int(
            unsatisfied["boundary_plus_multiple_roles"]
        ),
        "unsatisfied_boundary_total": len(unsatisfied_blocks),
        "unsatisfied_boundary_block_ids": unsatisfied_blocks,
    }


def _same_edge_conflict(
    edge: str,
    block_id: int,
    case: FloorSetCase,
    placements: dict[int, Placement],
    bbox: tuple[float, float, float, float],
) -> bool:
    requiring = [
        idx
        for idx in range(case.block_count)
        if idx in placements
        and edge in _boundary_edges(_block_code(case, idx, ConstraintColumns.BOUNDARY))
    ]
    if len(requiring) <= 1:
        return False
    capacity = (bbox[3] - bbox[1]) if edge in {"left", "right"} else (bbox[2] - bbox[0])
    intervals = {idx: _edge_interval(edge, placements[idx]) for idx in requiring}
    total_span = sum(max(hi - lo, 0.0) for lo, hi in intervals.values())
    own_lo, own_hi = intervals[block_id]
    overlaps = 0
    for other, (lo, hi) in intervals.items():
        if other == block_id:
            continue
        if max(own_lo, lo) < min(own_hi, hi):
            overlaps += 1
    return overlaps > 0 or total_span > capacity * 0.95


def _shape_mismatch(edge: str, box: Placement, bbox: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    span = h if edge in {"left", "right"} else w
    capacity = (bbox[3] - bbox[1]) if edge in {"left", "right"} else (bbox[2] - bbox[0])
    aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
    span_share = span / max(capacity, 1e-6)
    return bool(aspect > 8.0 or span_share < 0.02 or span_share > 0.70)


def classify_boundary_failures(
    case: FloorSetCase,
    pre_placements: dict[int, Placement],
    post_placements: dict[int, Placement],
    *,
    predicted_hull: PuzzleFrame | None,
    edge_owner_rows: list[dict[str, Any]] | None = None,
    candidate_coverage: CandidateEdgeCoverage | None = None,
) -> list[dict[str, Any]]:
    pre_bbox = _bbox_from_placements(pre_placements.values())
    post_bbox = _bbox_from_placements(post_placements.values())
    if post_bbox is None:
        return []
    owner_by_edge = {row["edge"]: row for row in edge_owner_rows or []}
    rows: list[dict[str, Any]] = []
    for idx in range(case.block_count):
        boundary_code = _block_code(case, idx, ConstraintColumns.BOUNDARY)
        if boundary_code == 0 or idx not in post_placements:
            continue
        post_sat, post_total = _bbox_boundary_satisfied_edges(
            boundary_code, post_placements[idx], post_bbox
        )
        if post_total == 0 or post_sat == post_total:
            continue
        required_edges = list(_boundary_edges(boundary_code))
        unsatisfied_edges = [
            edge
            for edge in required_edges
            if not _edge_owned(edge, post_placements[idx], post_bbox, 1e-4)
        ]
        pre_sat, pre_total = (
            _bbox_boundary_satisfied_edges(boundary_code, pre_placements[idx], pre_bbox)
            if pre_bbox is not None and idx in pre_placements
            else (0, post_total)
        )
        hull_sat, hull_total = (0, 0)
        if predicted_hull is not None and idx in pre_placements:
            hull_sat, hull_total = boundary_frame_satisfied_edges(
                boundary_code, pre_placements[idx], predicted_hull
            )
        flags = block_role_flags(case, idx)
        candidate_count = (
            candidate_coverage.total_required_count(idx, unsatisfied_edges)
            if candidate_coverage is not None
            else 0
        )
        final_owner_ids = sorted(
            {
                int(owner)
                for edge in unsatisfied_edges
                for owner in owner_by_edge.get(edge, {}).get("owner_block_ids", [])
            }
        )
        edge_stolen = any(
            bool(owner_by_edge.get(edge, {}).get("regular_or_nonboundary_stole_edge"))
            for edge in unsatisfied_edges
        )
        segment_conflict = any(
            _same_edge_conflict(edge, idx, case, post_placements, post_bbox)
            for edge in unsatisfied_edges
        )
        shape_mismatch = any(
            _shape_mismatch(edge, post_placements[idx], post_bbox) for edge in unsatisfied_edges
        )
        failure_reasons: list[str] = []
        if candidate_count == 0:
            failure_reasons.append("candidate_missing")
        block_moved_in_postprocess = (
            idx in pre_placements and pre_placements[idx] != post_placements[idx]
        )
        if (
            block_moved_in_postprocess
            and pre_total > 0
            and pre_sat == pre_total
            and post_sat < post_total
        ):
            failure_reasons.append("postprocess_changed")
        if hull_total > 0 and hull_sat == hull_total and post_sat < post_total:
            failure_reasons.append("on_predicted_hull_but_not_final_bbox")
        elif hull_total > 0 and hull_sat < hull_total:
            failure_reasons.append("not_on_predicted_hull")
        if edge_stolen:
            failure_reasons.append("edge_stolen_by_regular_or_nonboundary")
        if segment_conflict:
            failure_reasons.append("edge_segment_conflict")
        if bool(flags["is_grouping"]):
            failure_reasons.append("role_conflict_grouping")
        if bool(flags["is_mib"]):
            failure_reasons.append("role_conflict_mib")
        if shape_mismatch:
            failure_reasons.append("shape_mismatch")
        if candidate_count > 0 and hull_sat < hull_total:
            failure_reasons.append("candidate_not_selected")
        if float(flags["terminal_ratio"]) < 0.25 and not bool(flags["multiple_roles"]):
            failure_reasons.append("weak_boundary_or_ordering")

        priority = [
            "postprocess_changed",
            "on_predicted_hull_but_not_final_bbox",
            "edge_stolen_by_regular_or_nonboundary",
            "role_conflict_grouping",
            "role_conflict_mib",
            "edge_segment_conflict",
            "shape_mismatch",
            "candidate_missing",
            "candidate_not_selected",
            "not_on_predicted_hull",
            "weak_boundary_or_ordering",
        ]
        primary = next((reason for reason in priority if reason in failure_reasons), "unknown")
        rows.append(
            {
                "block_id": int(idx),
                "required_boundary_code": int(boundary_code),
                "required_boundary_type": "+".join(required_edges),
                "unsatisfied_edges": unsatisfied_edges,
                "failure_type": primary,
                "failure_reasons": failure_reasons,
                "candidate_count_for_required_edge": int(candidate_count),
                "selected_candidate_satisfied_predicted_hull": bool(
                    hull_total > 0 and hull_sat == hull_total
                ),
                "satisfied_pre_repair": bool(pre_total > 0 and pre_sat == pre_total),
                "satisfied_post_repair": False,
                "final_edge_owner_block_ids": final_owner_ids,
                "role_flags": flags,
            }
        )
    return rows


def compact_left_bottom_once(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    frame: PuzzleFrame | None = None,
    preserve_fixed_preplaced: bool = True,
) -> dict[int, Placement]:
    next_placements = dict(placements)

    def movable(block_id: int) -> bool:
        if not preserve_fixed_preplaced:
            return True
        return not (
            bool(_block_code(case, block_id, ConstraintColumns.FIXED))
            or bool(_block_code(case, block_id, ConstraintColumns.PREPLACED))
        )

    sep = 1e-4

    def overlaps(block_id: int, candidate: Placement) -> bool:
        x, y, w, h = candidate
        for other, box in next_placements.items():
            if other == block_id:
                continue
            ox, oy, ow, oh = box
            if max(x, ox) < min(x + w, ox + ow) - sep and max(y, oy) < min(y + h, oy + oh) - sep:
                return True
        return False

    def x_overlaps(candidate: Placement, other: Placement) -> bool:
        x, _y, w, _h = candidate
        ox, _oy, ow, _oh = other
        return max(x, ox) < min(x + w, ox + ow)

    def y_overlaps(candidate: Placement, other: Placement) -> bool:
        _x, y, _w, h = candidate
        _ox, oy, _ow, oh = other
        return max(y, oy) < min(y + h, oy + oh)

    bbox = _bbox_from_placements(next_placements.values())
    if bbox is None:
        return next_placements
    min_x = frame.xmin if frame is not None else bbox[0]
    min_y = frame.ymin if frame is not None else bbox[1]

    for idx in sorted(next_placements, key=lambda block: next_placements[block][0]):
        if not movable(idx):
            continue
        x, y, w, h = next_placements[idx]
        candidates = [min_x]
        for other, box in next_placements.items():
            if other == idx:
                continue
            ox, _oy, ow, _oh = box
            trial_same_y = (ox + ow, y, w, h)
            if ox + ow <= x + 1e-6 and y_overlaps(trial_same_y, box):
                candidates.append(ox + ow + sep)
        for new_x in sorted(candidates):
            candidate = (float(new_x), y, w, h)
            if new_x > x + 1e-6:
                continue
            if frame is not None and not frame.contains_box(candidate):
                continue
            if not overlaps(idx, candidate):
                next_placements[idx] = candidate
                break

    for idx in sorted(next_placements, key=lambda block: next_placements[block][1]):
        if not movable(idx):
            continue
        x, y, w, h = next_placements[idx]
        candidates = [min_y]
        for other, box in next_placements.items():
            if other == idx:
                continue
            _ox, oy, _ow, oh = box
            trial_same_x = (x, oy + oh, w, h)
            if oy + oh <= y + 1e-6 and x_overlaps(trial_same_x, box):
                candidates.append(oy + oh + sep)
        for new_y in sorted(candidates):
            candidate = (x, float(new_y), w, h)
            if new_y > y + 1e-6:
                continue
            if frame is not None and not frame.contains_box(candidate):
                continue
            if not overlaps(idx, candidate):
                next_placements[idx] = candidate
                break
    return next_placements


def compact_left_bottom(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    frame: PuzzleFrame | None = None,
    passes: int = 2,
    preserve_fixed_preplaced: bool = True,
) -> dict[int, Placement]:
    compacted = dict(placements)
    for _ in range(max(passes, 1)):
        updated = compact_left_bottom_once(
            case,
            compacted,
            frame=frame,
            preserve_fixed_preplaced=preserve_fixed_preplaced,
        )
        if updated == compacted:
            break
        compacted = updated
    return compacted


def summarize_failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["failure_type"]) for row in rows)
    return {key: int(value) for key, value in sorted(counts.items())}


def final_bbox_boundary_satisfaction(case: FloorSetCase, placements: dict[int, Placement]) -> float:
    return float(
        final_bbox_boundary_metrics(case, placements)["final_bbox_boundary_satisfaction_rate"]
    )
