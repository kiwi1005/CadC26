from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.research.boundary_failure_attribution import (
    block_role_flags,
    compact_left_bottom,
    final_bbox_edge_owner_audit,
)
from puzzleplace.research.compaction_alternatives import edge_aware_compaction
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _area,
    _bbox_boundary_satisfied_edges,
    _bbox_from_placements,
    _boundary_edges,
    final_bbox_boundary_metrics,
    frame_diagnostics,
    pin_box,
)

MoveMode = Literal["research", "safe"]

MOVE_LIBRARY = (
    "simple_compaction",
    "edge_aware_compaction",
    "soft_aspect_flip",
    "soft_shape_stretch",
    "mib_master_aspect_flip",
    "mib_master_edge_slot_shape",
    "group_template_rotate",
    "group_template_mirror",
    "group_boundary_touch_template",
    "local_region_repack",
    "boundary_edge_reassign",
    "cluster_split_or_two_lobe_repack",
)


@dataclass(frozen=True, slots=True)
class MoveCandidate:
    move_type: str
    target_blocks: tuple[int, ...]
    reason: str


def block_code(case: FloorSetCase, block_id: int, column: ConstraintColumns) -> int:
    return int(float(case.constraints[block_id, column].item()))


def is_fixed_or_preplaced(case: FloorSetCase, block_id: int) -> bool:
    return bool(block_code(case, block_id, ConstraintColumns.FIXED)) or bool(
        block_code(case, block_id, ConstraintColumns.PREPLACED)
    )


def hpwl_proxy(case: FloorSetCase, placements: dict[int, Placement]) -> float:
    total = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i not in placements or j not in placements:
            continue
        ix, iy, iw, ih = placements[i]
        jx, jy, jw, jh = placements[j]
        total += float(weight) * (
            abs((ix + iw / 2.0) - (jx + jw / 2.0))
            + abs((iy + ih / 2.0) - (jy + jh / 2.0))
        )
    return total


def soft_boundary_violations(case: FloorSetCase, placements: dict[int, Placement]) -> int:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return case.block_count
    violations = 0
    for idx, box in placements.items():
        code = block_code(case, idx, ConstraintColumns.BOUNDARY)
        if code == 0:
            continue
        sat, total = _bbox_boundary_satisfied_edges(code, box, bbox)
        violations += int(total > 0 and sat < total)
    return violations


def grouping_violation_count(case: FloorSetCase, placements: dict[int, Placement]) -> int:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in placements:
        group_id = block_code(case, idx, ConstraintColumns.CLUSTER)
        if group_id > 0:
            groups[group_id].append(idx)
    return sum(
        max(_shared_edge_components(members, placements) - 1, 0)
        for members in groups.values()
    )


def mib_violation_count(case: FloorSetCase, placements: dict[int, Placement]) -> int:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in placements:
        mib_id = block_code(case, idx, ConstraintColumns.MIB)
        if mib_id > 0:
            groups[mib_id].append(idx)
    violations = 0
    for members in groups.values():
        shapes = {
            (round(placements[idx][2], 4), round(placements[idx][3], 4))
            for idx in members
            if idx in placements
        }
        violations += int(len(shapes) > 1)
    return violations


def profile_case(
    case: FloorSetCase,
    placements: dict[int, Placement] | None = None,
    *,
    suite: str = "unassigned",
) -> dict[str, Any]:
    boundary_count = sum(
        int(block_code(case, idx, ConstraintColumns.BOUNDARY) != 0)
        for idx in range(case.block_count)
    )
    mib_count = sum(
        int(block_code(case, idx, ConstraintColumns.MIB) != 0)
        for idx in range(case.block_count)
    )
    grouping_count = sum(
        int(block_code(case, idx, ConstraintColumns.CLUSTER) != 0)
        for idx in range(case.block_count)
    )
    fixed_preplaced = sum(int(is_fixed_or_preplaced(case, idx)) for idx in range(case.block_count))
    external_ratios = [_external_ratio(case, idx) for idx in range(case.block_count)]
    areas = [float(value) for value in case.area_targets.tolist()]
    pins = pin_box(case)
    pin_w = max((pins[2] - pins[0]) if pins is not None else 0.0, 0.0)
    pin_h = max((pins[3] - pins[1]) if pins is not None else 0.0, 0.0)
    profile = {
        "case_id": str(case.case_id),
        "suite": suite,
        "n_blocks": case.block_count,
        "size_bucket": size_bucket(case.block_count),
        "boundary_density": boundary_count / max(case.block_count, 1),
        "mib_density": mib_count / max(case.block_count, 1),
        "grouping_density": grouping_count / max(case.block_count, 1),
        "external_ratio_mean": sum(external_ratios) / max(len(external_ratios), 1),
        "external_ratio_max": max(external_ratios, default=0.0),
        "pin_box_aspect": pin_w / max(pin_h, 1e-6) if pins is not None else 0.0,
        "pin_spread": math.sqrt(pin_w * pin_w + pin_h * pin_h),
        "area_skew": max(areas, default=0.0)
        / max(min((a for a in areas if a > 0), default=1.0), 1e-6),
        "fixed_preplaced_count": fixed_preplaced,
        "connectivity_modularity": _connectivity_modularity_proxy(case),
    }
    if placements is not None:
        boundary = final_bbox_boundary_metrics(case, placements)
        bbox = _bbox_from_placements(placements.values())
        frame_pressure = _area(bbox) / max(float(case.area_targets.sum().item()), 1e-6)
        profile.update(
            {
                "baseline_boundary_failure_rate": 1.0
                - float(boundary["final_bbox_boundary_satisfaction_rate"]),
                "baseline_bbox_pressure": frame_pressure,
            }
        )
    return profile


def size_bucket(block_count: int) -> str:
    if block_count <= 40:
        return "small"
    if block_count <= 70:
        return "medium"
    if block_count <= 100:
        return "large"
    return "xl"


def build_case_suite(
    cases: list[FloorSetCase],
    *,
    diagnostic_count: int = 20,
    holdout_count: int = 20,
) -> dict[str, Any]:
    max_count = min(len(cases), diagnostic_count + holdout_count)
    diagnostic_ids = list(range(min(diagnostic_count, max_count)))
    holdout_ids = list(range(min(diagnostic_count, max_count), max_count))
    bucket_counts: dict[str, Counter[str]] = {
        "diagnostic": Counter(size_bucket(cases[idx].block_count) for idx in diagnostic_ids),
        "holdout": Counter(size_bucket(cases[idx].block_count) for idx in holdout_ids),
    }
    return {
        "diagnostic_case_ids": diagnostic_ids,
        "holdout_case_ids": holdout_ids,
        "case_count": max_count,
        "diagnostic_count": len(diagnostic_ids),
        "holdout_count": len(holdout_ids),
        "size_bucket_counts": {
            suite: {key: int(value) for key, value in sorted(counter.items())}
            for suite, counter in bucket_counts.items()
        },
        "note": (
            "Validation loader order currently provides block counts 21..60 for the "
            "default 40-case suite; large/xl buckets are tracked but unavailable in "
            "this local slice."
        ),
    }


def target_pool(
    case: FloorSetCase,
    placements: dict[int, Placement],
    failure_rows: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[int]:
    scores: Counter[int] = Counter()
    for row in failure_rows:
        block_id = int(row.get("block_id", -1))
        if block_id >= 0:
            scores[block_id] += 10
        for owner in row.get("final_edge_owner_block_ids", []):
            scores[int(owner)] += 4
    for edge_row in final_bbox_edge_owner_audit(case, placements):
        if edge_row.get("regular_or_nonboundary_stole_edge"):
            for owner in edge_row.get("owner_block_ids", []):
                scores[int(owner)] += 5
            for req in edge_row.get("unsatisfied_required_block_ids", []):
                scores[int(req)] += 7
    for idx in range(case.block_count):
        if block_code(case, idx, ConstraintColumns.BOUNDARY) != 0:
            scores[idx] += 1
    return [idx for idx, _score in scores.most_common(top_k) if idx in placements]


def generate_move_candidates(
    case: FloorSetCase,
    placements: dict[int, Placement],
    failure_rows: list[dict[str, Any]],
    *,
    top_k_targets: int = 5,
    top_m_moves: int = 12,
) -> list[MoveCandidate]:
    targets = target_pool(case, placements, failure_rows, top_k=top_k_targets)
    rows = [
        MoveCandidate("simple_compaction", tuple(targets), "global_baseline_alternative"),
        MoveCandidate(
            "edge_aware_compaction",
            tuple(targets),
            "boundary_owner_preserving_compaction",
        ),
    ]
    per_target_moves = [
        "soft_aspect_flip",
        "soft_shape_stretch",
        "mib_master_aspect_flip",
        "mib_master_edge_slot_shape",
        "group_template_rotate",
        "group_template_mirror",
        "group_boundary_touch_template",
        "local_region_repack",
        "boundary_edge_reassign",
        "cluster_split_or_two_lobe_repack",
    ][:top_m_moves]
    for target in targets:
        for move_type in per_target_moves:
            rows.append(MoveCandidate(move_type, (target,), "attribution_target"))
    return rows


def evaluate_move(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    move: MoveCandidate,
    *,
    mode: MoveMode = "research",
) -> dict[str, Any]:
    before = layout_metrics(case, placements, frame)
    t0 = time.perf_counter()
    alternative, generation_reasons, candidate_count = apply_move(case, placements, frame, move)
    generation_ms = (time.perf_counter() - t0) * 1000.0
    repair_ms = 0.0
    t1 = time.perf_counter()
    after = layout_metrics(case, alternative, frame)
    eval_ms = (time.perf_counter() - t1) * 1000.0
    deltas = metric_deltas(before, after)
    rejected = rejection_reasons(before, after, generation_reasons, mode=mode)
    accepted = not rejected
    improvement = improvement_score(deltas)
    return {
        "case_id": str(case.case_id),
        "move_type": move.move_type,
        "target_blocks": list(move.target_blocks),
        "target_roles": {
            str(idx): block_role_flags(case, idx) for idx in move.target_blocks
        },
        "before_metrics": before,
        "after_metrics": after,
        "accepted": accepted,
        "rejected_reason": rejected,
        "selection_reason": "accepted_by_safe_gate" if accepted else "rejected_by_gate",
        "boundary_delta": deltas["boundary_delta"],
        "bbox_delta": deltas["bbox_delta"],
        "hpwl_delta": deltas["hpwl_delta"],
        "soft_delta": deltas["soft_delta"],
        "grouping_delta": deltas["grouping_delta"],
        "mib_delta": deltas["mib_delta"],
        "hard_feasible": bool(after["hard_feasible"]),
        "frame_protrusion": float(after["frame_protrusion"]),
        "generation_time_ms": generation_ms,
        "repair_time_ms": repair_ms,
        "eval_time_ms": eval_ms,
        "num_candidates_evaluated": candidate_count,
        "improvement_per_ms": improvement / max(generation_ms + repair_ms + eval_ms, 1e-6),
    }


def apply_move(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    move: MoveCandidate,
) -> tuple[dict[int, Placement], list[str], int]:
    if move.move_type == "simple_compaction":
        return compact_left_bottom(case, placements, frame=frame, passes=4), [], 1
    if move.move_type == "edge_aware_compaction":
        return edge_aware_compaction(case, placements, frame=frame, passes=4).placements, [], 1
    if not move.target_blocks:
        return dict(placements), ["missing_target"], 0
    target = int(move.target_blocks[0])
    if target not in placements:
        return dict(placements), ["target_not_placed"], 0
    if move.move_type == "soft_aspect_flip":
        return _soft_aspect_flip(case, placements, frame, target)
    if move.move_type == "soft_shape_stretch":
        return _soft_shape_stretch(case, placements, frame, target)
    if move.move_type == "mib_master_aspect_flip":
        return _mib_master_aspect_flip(case, placements, frame, target)
    if move.move_type == "mib_master_edge_slot_shape":
        return _mib_master_edge_slot_shape(case, placements, frame, target)
    if move.move_type == "group_template_rotate":
        return _group_transform(case, placements, frame, target, transform="rotate")
    if move.move_type == "group_template_mirror":
        return _group_transform(case, placements, frame, target, transform="mirror")
    if move.move_type == "group_boundary_touch_template":
        return _group_boundary_touch(case, placements, frame, target)
    if move.move_type == "local_region_repack":
        return _local_region_repack(case, placements, frame, target)
    if move.move_type == "boundary_edge_reassign":
        return _boundary_edge_reassign(case, placements, frame, target)
    if move.move_type == "cluster_split_or_two_lobe_repack":
        return _cluster_two_lobe(case, placements, frame, target)
    return dict(placements), ["unknown_move_type"], 0


def layout_metrics(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, Any]:
    legality = summarize_hard_legality(case, _positions_list(placements, case.block_count))
    boundary = final_bbox_boundary_metrics(case, placements)
    frame_metrics = frame_diagnostics(case, placements, frame)
    return {
        "boundary_satisfaction_rate": float(boundary["final_bbox_boundary_satisfaction_rate"]),
        "bbox_area": _area(_bbox_from_placements(placements.values())),
        "hpwl_proxy": hpwl_proxy(case, placements),
        "soft_boundary_violations": soft_boundary_violations(case, placements),
        "grouping_violations": grouping_violation_count(case, placements),
        "mib_violations": mib_violation_count(case, placements),
        "hard_feasible": bool(legality.is_feasible),
        "frame_protrusion": float(frame_metrics["max_protrusion_distance"]),
        "frame_violations": int(frame_metrics["num_frame_violations"]),
        "outside_frame_area_ratio": float(frame_metrics["outside_frame_area_ratio"]),
        "mean_displacement": 0.0,
    }


def metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "boundary_delta": float(after["boundary_satisfaction_rate"])
        - float(before["boundary_satisfaction_rate"]),
        "bbox_delta": float(after["bbox_area"]) - float(before["bbox_area"]),
        "hpwl_delta": float(after["hpwl_proxy"]) - float(before["hpwl_proxy"]),
        "soft_delta": float(after["soft_boundary_violations"])
        - float(before["soft_boundary_violations"]),
        "grouping_delta": float(after["grouping_violations"])
        - float(before["grouping_violations"]),
        "mib_delta": float(after["mib_violations"]) - float(before["mib_violations"]),
    }


def rejection_reasons(
    before: dict[str, Any],
    after: dict[str, Any],
    generation_reasons: list[str],
    *,
    mode: MoveMode,
) -> list[str]:
    reasons = list(generation_reasons)
    if not after["hard_feasible"]:
        reasons.append("hard_infeasible")
    if float(after["frame_protrusion"]) > 1e-4 or int(after["frame_violations"]) > 0:
        reasons.append("frame_protrusion")
    deltas = metric_deltas(before, after)
    if mode == "safe":
        if all(abs(value) <= 1e-12 for value in deltas.values()):
            reasons.append("no_effect")
        if deltas["boundary_delta"] < -1e-9:
            reasons.append("boundary_worse")
        if deltas["bbox_delta"] > max(1.0, 0.05 * float(before["bbox_area"])):
            reasons.append("bbox_worse")
        if deltas["hpwl_delta"] > max(1.0, 0.25 * float(before["hpwl_proxy"])):
            reasons.append("hpwl_worse")
        if deltas["soft_delta"] > 1:
            reasons.append("soft_worse")
    return sorted(set(reasons))


def select_case_alternative(
    rows: list[dict[str, Any]],
    *,
    mode: MoveMode = "safe",
) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    original = _original_selection_payload(rows)
    if not accepted:
        return {**original, "selection_reason": "no_accepted_move"}

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            -float(row["boundary_delta"]),
            float(row["bbox_delta"]),
            float(row["hpwl_delta"]),
            float(row["soft_delta"]),
            float(row["generation_time_ms"] + row["eval_time_ms"]),
        )

    selected = min(accepted, key=key)
    if mode == "safe" and improvement_score(selected) <= 0.0:
        return {**original, "selection_reason": "accepted_moves_not_net_positive"}
    return {
        "case_id": selected["case_id"],
        "selected_move_type": selected["move_type"],
        "selected_target_blocks": selected["target_blocks"],
        "selection_reason": selected["selection_reason"],
        "boundary_delta": selected["boundary_delta"],
        "bbox_delta": selected["bbox_delta"],
        "hpwl_delta": selected["hpwl_delta"],
        "soft_delta": selected["soft_delta"],
        "grouping_delta": selected["grouping_delta"],
        "mib_delta": selected["mib_delta"],
        "hard_feasible": selected["hard_feasible"],
        "frame_protrusion": selected["frame_protrusion"],
        "runtime_ms": selected["generation_time_ms"]
        + selected["repair_time_ms"]
        + selected["eval_time_ms"],
    }


def move_cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_move: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_move[str(row["move_type"])].append(row)
    summary: dict[str, Any] = {}
    for move_type, group in sorted(by_move.items()):
        total_ms = [
            float(row["generation_time_ms"] + row["repair_time_ms"] + row["eval_time_ms"])
            for row in group
        ]
        summary[move_type] = {
            "count": len(group),
            "accepted_count": sum(int(row["accepted"]) for row in group),
            "mean_total_ms": sum(total_ms) / max(len(total_ms), 1),
            "mean_improvement_per_ms": sum(float(row["improvement_per_ms"]) for row in group)
            / max(len(group), 1),
            "mean_boundary_delta": sum(float(row["boundary_delta"]) for row in group)
            / max(len(group), 1),
            "mean_bbox_delta": sum(float(row["bbox_delta"]) for row in group)
            / max(len(group), 1),
            "mean_hpwl_delta": sum(float(row["hpwl_delta"]) for row in group)
            / max(len(group), 1),
        }
    return summary


def profile_summary(
    profiles: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_by_case = {str(row["case_id"]): row for row in profiles}
    by_suite: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selections:
        case_id = str(row["case_id"]).replace("validation-", "")
        profile = profile_by_case.get(case_id) or profile_by_case.get(str(row["case_id"]), {})
        suite = str(profile.get("suite", "unknown"))
        bucket = str(profile.get("size_bucket", "unknown"))
        move_type = str(row["selected_move_type"])
        by_suite[suite][move_type] += 1
        by_bucket[bucket][move_type] += 1
    return {
        "preferred_moves_by_suite": {
            key: dict(value) for key, value in sorted(by_suite.items())
        },
        "preferred_moves_by_size_bucket": {
            key: dict(value) for key, value in sorted(by_bucket.items())
        },
    }


def improvement_score(row_or_deltas: dict[str, Any]) -> float:
    boundary = float(row_or_deltas.get("boundary_delta", 0.0))
    bbox = float(row_or_deltas.get("bbox_delta", 0.0))
    hpwl = float(row_or_deltas.get("hpwl_delta", 0.0))
    soft = float(row_or_deltas.get("soft_delta", 0.0))
    return 100.0 * boundary - 0.001 * max(bbox, 0.0) - 0.01 * max(hpwl, 0.0) - soft


def _original_selection_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_id = rows[0]["case_id"] if rows else "unknown"
    return {
        "case_id": case_id,
        "selected_move_type": "original",
        "selected_target_blocks": [],
        "boundary_delta": 0.0,
        "bbox_delta": 0.0,
        "hpwl_delta": 0.0,
        "soft_delta": 0.0,
        "grouping_delta": 0.0,
        "mib_delta": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
        "runtime_ms": 0.0,
    }


def _external_ratio(case: FloorSetCase, block_id: int) -> float:
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


def _connectivity_modularity_proxy(case: FloorSetCase) -> float:
    group_edges = 0
    total_edges = 0
    for src, dst, _weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i < 0 or j < 0 or i >= case.block_count or j >= case.block_count:
            continue
        total_edges += 1
        if block_code(case, i, ConstraintColumns.CLUSTER) == block_code(
            case, j, ConstraintColumns.CLUSTER
        ) and block_code(case, i, ConstraintColumns.CLUSTER) > 0:
            group_edges += 1
    return group_edges / max(total_edges, 1)


def _positions_list(placements: dict[int, Placement], block_count: int) -> list[Placement]:
    return [placements[idx] for idx in range(block_count)]


def _soft_aspect_flip(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    if is_fixed_or_preplaced(case, target):
        return dict(placements), ["fixed_or_preplaced_exact"], 0
    x, y, w, h = placements[target]
    candidate = (x + (w - h) / 2.0, y + (h - w) / 2.0, h, w)
    return _try_single_box(placements, frame, target, candidate), [], 1


def _soft_shape_stretch(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    if is_fixed_or_preplaced(case, target):
        return dict(placements), ["fixed_or_preplaced_exact"], 0
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return dict(placements), ["missing_bbox"], 0
    edges = _boundary_edges(block_code(case, target, ConstraintColumns.BOUNDARY))
    area = max(float(case.area_targets[target].item()), 1e-6)
    ratio = math.exp(-1.0) if any(edge in {"left", "right"} for edge in edges) else math.exp(1.0)
    width = math.sqrt(area * ratio)
    height = math.sqrt(area / ratio)
    edge = edges[0] if edges else "center"
    candidate = _align_shape_to_edge(edge, placements[target], width, height, bbox)
    return _try_single_box(placements, frame, target, candidate), [], 1


def _mib_master_aspect_flip(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    members = _mib_members(case, target, placements)
    if not members:
        return dict(placements), ["target_not_mib"], 0
    updated = dict(placements)
    for idx in members:
        if is_fixed_or_preplaced(case, idx):
            return dict(placements), ["fixed_or_preplaced_mib_member"], len(members)
        x, y, w, h = updated[idx]
        updated[idx] = (x + (w - h) / 2.0, y + (h - w) / 2.0, h, w)
    return _validate_candidate_layout(placements, updated, frame), [], len(members)


def _mib_master_edge_slot_shape(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    members = _mib_members(case, target, placements)
    if not members:
        return dict(placements), ["target_not_mib"], 0
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return dict(placements), ["missing_bbox"], 0
    edges = _boundary_edges(block_code(case, target, ConstraintColumns.BOUNDARY))
    edge = edges[0] if edges else "left"
    updated = dict(placements)
    for idx in members[:3]:
        if is_fixed_or_preplaced(case, idx):
            continue
        area = max(float(case.area_targets[idx].item()), 1e-6)
        width, height = _edge_slot_shape(area, edge, bbox)
        updated[idx] = _align_shape_to_edge(edge, updated[idx], width, height, bbox)
    return _validate_candidate_layout(placements, updated, frame), [], min(len(members), 3)


def _group_transform(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
    *,
    transform: str,
) -> tuple[dict[int, Placement], list[str], int]:
    members = _group_members(case, target, placements)
    if not members:
        return dict(placements), ["target_not_grouping"], 0
    if any(is_fixed_or_preplaced(case, idx) for idx in members):
        return dict(placements), ["fixed_or_preplaced_group_member"], len(members)
    group_bbox = _bbox_from_placements(placements[idx] for idx in members)
    if group_bbox is None:
        return dict(placements), ["missing_group_bbox"], 0
    cx, cy = (group_bbox[0] + group_bbox[2]) / 2.0, (group_bbox[1] + group_bbox[3]) / 2.0
    updated = dict(placements)
    for idx in members:
        x, y, w, h = placements[idx]
        bx, by = x + w / 2.0, y + h / 2.0
        if transform == "mirror":
            nx, ny = 2.0 * cx - bx, by
        else:
            nx, ny = cx - (by - cy), cy + (bx - cx)
        updated[idx] = (nx - w / 2.0, ny - h / 2.0, w, h)
    return _validate_candidate_layout(placements, updated, frame), [], len(members)


def _group_boundary_touch(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    members = _group_members(case, target, placements)
    if not members:
        return dict(placements), ["target_not_grouping"], 0
    bbox = _bbox_from_placements(placements.values())
    group_bbox = _bbox_from_placements(placements[idx] for idx in members)
    if bbox is None or group_bbox is None:
        return dict(placements), ["missing_bbox"], 0
    edges = _boundary_edges(block_code(case, target, ConstraintColumns.BOUNDARY))
    edge = edges[0] if edges else "left"
    dx = dy = 0.0
    if edge == "left":
        dx = bbox[0] - group_bbox[0]
    elif edge == "right":
        dx = bbox[2] - group_bbox[2]
    elif edge == "bottom":
        dy = bbox[1] - group_bbox[1]
    elif edge == "top":
        dy = bbox[3] - group_bbox[3]
    updated = dict(placements)
    for idx in members:
        if is_fixed_or_preplaced(case, idx):
            return dict(placements), ["fixed_or_preplaced_group_member"], len(members)
        x, y, w, h = placements[idx]
        updated[idx] = (x + dx, y + dy, w, h)
    return _validate_candidate_layout(placements, updated, frame), [], len(members)


def _local_region_repack(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    local = set(_local_region(case, placements, target))
    compacted = compact_left_bottom(case, placements, frame=frame, passes=2)
    updated = dict(placements)
    for idx in local:
        if idx in compacted and not is_fixed_or_preplaced(case, idx):
            updated[idx] = compacted[idx]
    return _validate_candidate_layout(placements, updated, frame), [], len(local)


def _boundary_edge_reassign(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    if is_fixed_or_preplaced(case, target):
        return dict(placements), ["fixed_or_preplaced_exact"], 0
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return dict(placements), ["missing_bbox"], 0
    edges = _boundary_edges(block_code(case, target, ConstraintColumns.BOUNDARY))
    if not edges:
        return dict(placements), ["target_not_boundary"], 0
    x, y, w, h = placements[target]
    candidate = _align_shape_to_edge(edges[0], (x, y, w, h), w, h, bbox)
    return _try_single_box(placements, frame, target, candidate), [], 1


def _cluster_two_lobe(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
) -> tuple[dict[int, Placement], list[str], int]:
    members = _group_members(case, target, placements)
    if len(members) < 2:
        return dict(placements), ["target_group_too_small"], 0
    if any(is_fixed_or_preplaced(case, idx) for idx in members):
        return dict(placements), ["fixed_or_preplaced_group_member"], len(members)
    group_bbox = _bbox_from_placements(placements[idx] for idx in members)
    if group_bbox is None:
        return dict(placements), ["missing_group_bbox"], 0
    left, right = members[::2], members[1::2]
    updated = dict(placements)
    y_left = group_bbox[1]
    y_right = group_bbox[1]
    x_left = group_bbox[0]
    x_right = (group_bbox[0] + group_bbox[2]) / 2.0
    for idx in left:
        _x, _y, w, h = placements[idx]
        updated[idx] = (x_left, y_left, w, h)
        y_left += h + 1e-4
    for idx in right:
        _x, _y, w, h = placements[idx]
        updated[idx] = (x_right, y_right, w, h)
        y_right += h + 1e-4
    return _validate_candidate_layout(placements, updated, frame), [], len(members)


def _try_single_box(
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    target: int,
    candidate: Placement,
) -> dict[int, Placement]:
    updated = dict(placements)
    updated[target] = candidate
    return _validate_candidate_layout(placements, updated, frame)


def _validate_candidate_layout(
    original: dict[int, Placement],
    candidate: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[int, Placement]:
    for box in candidate.values():
        if not frame.contains_box(box):
            return dict(original)
    if _has_overlap(candidate):
        return dict(original)
    return candidate


def _has_overlap(placements: dict[int, Placement], *, sep: float = 1e-4) -> bool:
    rows = list(placements.items())
    for pos, (_idx, a) in enumerate(rows):
        ax, ay, aw, ah = a
        for _other, b in rows[pos + 1 :]:
            bx, by, bw, bh = b
            x_overlap = max(ax, bx) < min(ax + aw, bx + bw) - sep
            y_overlap = max(ay, by) < min(ay + ah, by + bh) - sep
            if x_overlap and y_overlap:
                return True
    return False


def _align_shape_to_edge(
    edge: str,
    old_box: Placement,
    width: float,
    height: float,
    bbox: tuple[float, float, float, float],
) -> Placement:
    x, y, old_w, old_h = old_box
    cx = x + old_w / 2.0
    cy = y + old_h / 2.0
    if edge == "left":
        return bbox[0], cy - height / 2.0, width, height
    if edge == "right":
        return bbox[2] - width, cy - height / 2.0, width, height
    if edge == "bottom":
        return cx - width / 2.0, bbox[1], width, height
    if edge == "top":
        return cx - width / 2.0, bbox[3] - height, width, height
    return cx - width / 2.0, cy - height / 2.0, width, height


def _edge_slot_shape(
    area: float,
    edge: str,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    if edge in {"left", "right"}:
        height = max((bbox[3] - bbox[1]) * 0.25, math.sqrt(area))
        return area / max(height, 1e-6), height
    width = max((bbox[2] - bbox[0]) * 0.25, math.sqrt(area))
    return width, area / max(width, 1e-6)


def _mib_members(
    case: FloorSetCase,
    target: int,
    placements: dict[int, Placement],
) -> list[int]:
    mib_id = block_code(case, target, ConstraintColumns.MIB)
    if mib_id <= 0:
        return []
    return [
        idx
        for idx in placements
        if block_code(case, idx, ConstraintColumns.MIB) == mib_id
    ]


def _group_members(
    case: FloorSetCase,
    target: int,
    placements: dict[int, Placement],
) -> list[int]:
    group_id = block_code(case, target, ConstraintColumns.CLUSTER)
    if group_id <= 0:
        return []
    return [
        idx
        for idx in placements
        if block_code(case, idx, ConstraintColumns.CLUSTER) == group_id
    ]


def _local_region(
    case: FloorSetCase,
    placements: dict[int, Placement],
    target: int,
) -> list[int]:
    region = {target}
    owners = final_bbox_edge_owner_audit(case, placements)
    required = set(_boundary_edges(block_code(case, target, ConstraintColumns.BOUNDARY)))
    for row in owners:
        if row["edge"] in required:
            region.update(int(value) for value in row.get("owner_block_ids", []))
    region.update(_mib_members(case, target, placements))
    region.update(_group_members(case, target, placements))
    neighbors: defaultdict[int, float] = defaultdict(float)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i == target and j in placements:
            neighbors[j] += abs(float(weight))
        if j == target and i in placements:
            neighbors[i] += abs(float(weight))
    region.update(
        idx
        for idx, _score in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    return [idx for idx in sorted(region) if idx in placements]


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
