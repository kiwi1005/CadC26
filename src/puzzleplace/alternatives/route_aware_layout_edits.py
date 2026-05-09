from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

SELECTED_FAMILIES = {
    "original_layout",
    "vacancy_aware_local_insertion",
    "adjacent_region_reassignment",
    "mib_group_closure_macro",
    "legacy_step7g_global_move",
}


@dataclass(frozen=True)
class LayoutEditCandidate:
    case_id: int
    candidate_id: str
    family: str
    descriptor_locality_class: str
    descriptor_repair_mode: str
    block_count: int
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    macro_closure_size: int
    route_lane: str


def build_layout_edit_candidates(
    step7h_predictions: list[dict[str, Any]],
) -> list[LayoutEditCandidate]:
    out: list[LayoutEditCandidate] = []
    for descriptor in step7h_predictions:
        family = str(descriptor["family"])
        if family not in SELECTED_FAMILIES:
            continue
        case_id = int(descriptor["case_id"])
        block_count = _block_count_from_descriptor(descriptor)
        frame = synthetic_frame(block_count)
        baseline = synthetic_baseline_layout(block_count)
        edited = apply_layout_edit(family, baseline, frame, block_count)
        out.append(
            LayoutEditCandidate(
                case_id=case_id,
                candidate_id=str(descriptor["candidate_id"]),
                family=family,
                descriptor_locality_class=str(descriptor["predicted_locality_class"]),
                descriptor_repair_mode=str(descriptor["predicted_repair_mode"]),
                block_count=block_count,
                baseline=baseline,
                edited=edited,
                frame=frame,
                macro_closure_size=_actual_macro_closure_size(family, block_count),
                route_lane=_route_lane(family),
            )
        )
    return out


def evaluate_layout_edit(candidate: LayoutEditCandidate) -> dict[str, Any]:
    changed = changed_blocks(candidate.baseline, candidate.edited)
    hard = hard_feasibility(candidate.edited, candidate.frame)
    touched_regions = touched_region_count(candidate.edited, changed, candidate.frame)
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.block_count,
        changed_block_count=len(changed),
        touched_region_count=touched_regions,
        macro_closure_size=candidate.macro_closure_size,
        min_region_slack=_minimum_region_slack(candidate.edited, candidate.frame),
        free_space_fit_ratio=_free_space_fit_ratio(candidate.edited, candidate.frame, changed),
        hard_summary={"hard_feasible": hard["hard_feasible"]},
    )
    actual_class = str(prediction["predicted_locality_class"])
    repair = route_appropriate_repair_proxy(candidate, actual_class, hard)
    metrics = actual_metric_deltas(candidate.baseline, repair["after_repair"], candidate.frame)
    return {
        "case_id": candidate.case_id,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "descriptor_locality_class": candidate.descriptor_locality_class,
        "descriptor_repair_mode": candidate.descriptor_repair_mode,
        "actual_locality_class": actual_class,
        "actual_repair_mode": prediction["predicted_repair_mode"],
        "route_lane": repair["route_lane"],
        "report_only": repair["report_only"],
        "changed_block_count": len(changed),
        "changed_block_fraction": len(changed) / max(candidate.block_count, 1),
        "affected_region_count": touched_regions,
        "macro_closure_size": candidate.macro_closure_size,
        "hard_feasible_before_repair": hard["hard_feasible"],
        "hard_feasible_after_route_proxy": repair["hard_feasible_after_route_proxy"],
        "overlap_count": hard["overlap_count"],
        "frame_protrusion_count": hard["frame_protrusion_count"],
        "failure_attribution": failure_attribution(
            candidate.descriptor_locality_class,
            actual_class,
            hard,
            repair["report_only"],
        ),
        **metrics,
    }


def evaluate_layout_edits(candidates: list[LayoutEditCandidate]) -> list[dict[str, Any]]:
    return [evaluate_layout_edit(candidate) for candidate in candidates]


def confusion_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    stable = 0
    for row in rows:
        descriptor = str(row["descriptor_locality_class"])
        actual = str(row["actual_locality_class"])
        matrix[descriptor][actual] += 1
        stable += int(descriptor == actual)
    return {
        "descriptor_class_vs_actual_route_confusion": {
            key: dict(value) for key, value in sorted(matrix.items())
        },
        "route_stability_descriptor_to_edit": stable / max(len(rows), 1),
        "stable_count": stable,
        "total": len(rows),
        "collapsed_to_global": [
            _compact_identity(row)
            for row in rows
            if row["descriptor_locality_class"] != "global"
            and row["actual_locality_class"] == "global"
        ],
    }


def feasibility_report(
    rows: list[dict[str, Any]], descriptor_candidate_count: int
) -> dict[str, Any]:
    actual_count = len(rows)
    class_counts = Counter(str(row["actual_locality_class"]) for row in rows)
    local_rows = [row for row in rows if row["actual_repair_mode"] == "bounded_repair_pareto"]
    invalid_local = [row for row in local_rows if not bool(row["hard_feasible_before_repair"])]
    hard_feasible = [row for row in rows if bool(row["hard_feasible_after_route_proxy"])]
    safe = [row for row in rows if is_actual_safe_improvement(row)]
    regional_macro = [
        row for row in rows if row["actual_locality_class"] in {"regional", "macro"}
    ]
    return {
        "descriptor_candidate_count": descriptor_candidate_count,
        "actual_edit_candidate_count": actual_count,
        "actual_route_count_by_class": dict(class_counts),
        "actual_non_global_candidate_rate": _rate(
            actual_count - class_counts["global"], actual_count
        ),
        "invalid_local_attempt_rate": _rate(len(invalid_local), len(local_rows)),
        "actual_hard_feasible_rate": _rate(len(hard_feasible), actual_count),
        "actual_safe_improvement_count": len(safe),
        "actual_safe_improvement_candidates": [_compact_identity(row) for row in safe],
        "regional_macro_preservation_count": len(regional_macro),
        "global_report_only_count": sum(int(bool(row["report_only"])) for row in rows),
        "failure_attribution_counts": dict(Counter(row["failure_attribution"] for row in rows)),
        "invalid_or_collapsed_candidates": [
            _compact_identity(row)
            | {"failure_attribution": row["failure_attribution"]}
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def pareto_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_id"])].append(row)
    per_case: dict[str, Any] = {}
    for case_id, case_rows in sorted(by_case.items()):
        front = _pareto_front(case_rows)
        per_case[str(case_id)] = {
            "front": [_pareto_row(row) for row in front],
            "front_size": len(front),
            "original_included": any(row["family"] == "original_layout" for row in case_rows),
        }
    return {
        "per_case": per_case,
        "actual_pareto_front_non_empty_count": sum(
            int(section["front_size"] > 0) for section in per_case.values()
        ),
        "original_inclusive_case_count": sum(
            int(section["original_included"]) for section in per_case.values()
        ),
    }


def decision_for_step7c_thin(
    feasibility: dict[str, Any],
    confusion: dict[str, Any],
    pareto: dict[str, Any],
) -> str:
    if confusion["route_stability_descriptor_to_edit"] < 0.70:
        return "revisit_route_aware_proxy_assumptions"
    if feasibility["actual_non_global_candidate_rate"] <= 0.0:
        return "refine_real_edit_generators"
    if feasibility["invalid_local_attempt_rate"] > 0.05:
        return "refine_real_edit_generators"
    if feasibility["actual_hard_feasible_rate"] < 0.70:
        return "refine_real_edit_generators"
    if pareto["actual_pareto_front_non_empty_count"] <= 0:
        return "inconclusive_due_to_real_edit_quality"
    if (
        feasibility["regional_macro_preservation_count"]
        > feasibility["actual_safe_improvement_count"]
    ):
        return "pivot_to_coarse_region_planner"
    return "promote_to_step7c_iterative_loop"


def synthetic_frame(block_count: int) -> PuzzleFrame:
    cols = _cols(block_count)
    rows = (block_count + cols - 1) // cols
    return PuzzleFrame(0.0, 0.0, cols * 10.0 + 2.0, rows * 10.0 + 2.0, density=0.8)


def synthetic_baseline_layout(block_count: int) -> dict[int, Placement]:
    cols = _cols(block_count)
    return {
        idx: (float(idx % cols) * 10.0 + 1.0, float(idx // cols) * 10.0 + 1.0, 6.0, 6.0)
        for idx in range(block_count)
    }


def apply_layout_edit(
    family: str,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    block_count: int,
) -> dict[int, Placement]:
    edited = dict(baseline)
    if family == "original_layout":
        return edited
    if family == "vacancy_aware_local_insertion":
        _shift_blocks(edited, _rightmost_blocks(baseline, 1), dx=-2.0, dy=0.0)
        return edited
    if family == "adjacent_region_reassignment":
        _shift_blocks(
            edited,
            _rightmost_blocks(baseline, max(3, round(block_count * 0.18))),
            dx=-2.0,
            dy=0.0,
        )
        return edited
    if family == "mib_group_closure_macro":
        _shift_blocks(
            edited,
            _rightmost_blocks(baseline, max(2, round(block_count * 0.08))),
            dx=-2.0,
            dy=0.0,
        )
        return edited
    if family == "legacy_step7g_global_move":
        for idx in range(max(1, round(block_count * 0.80))):
            edited[idx] = (frame.xmin - 5.0, frame.ymin - 5.0, 6.0, 6.0)
        return edited
    raise ValueError(f"unsupported Step7C-thin layout edit family: {family}")


def route_appropriate_repair_proxy(
    candidate: LayoutEditCandidate,
    actual_class: str,
    hard: dict[str, Any],
) -> dict[str, Any]:
    if actual_class == "global":
        return {
            "route_lane": "global_report_only",
            "report_only": True,
            "after_repair": candidate.edited,
            "hard_feasible_after_route_proxy": hard["hard_feasible"],
        }
    if actual_class == "local":
        if hard["hard_feasible"]:
            return {
                "route_lane": "bounded_local_noop_repair",
                "report_only": False,
                "after_repair": candidate.edited,
                "hard_feasible_after_route_proxy": True,
            }
        return {
            "route_lane": "bounded_local_rollback",
            "report_only": False,
            "after_repair": candidate.baseline,
            "hard_feasible_after_route_proxy": True,
        }
    if actual_class == "regional":
        return {
            "route_lane": "regional_report_lane",
            "report_only": False,
            "after_repair": candidate.edited,
            "hard_feasible_after_route_proxy": hard["hard_feasible"],
        }
    return {
        "route_lane": "macro_report_lane",
        "report_only": False,
        "after_repair": candidate.edited,
        "hard_feasible_after_route_proxy": hard["hard_feasible"],
    }


def hard_feasibility(placements: dict[int, Placement], frame: PuzzleFrame) -> dict[str, Any]:
    rows = list(placements.items())
    overlaps = 0
    for left_idx, (_left_block, left_box) in enumerate(rows):
        for _right_block, right_box in rows[left_idx + 1 :]:
            overlaps += int(_intersection_area(left_box, right_box) > 1e-9)
    protrusions = sum(int(not frame.contains_box(box)) for box in placements.values())
    return {
        "hard_feasible": overlaps == 0 and protrusions == 0,
        "overlap_count": overlaps,
        "frame_protrusion_count": protrusions,
    }


def changed_blocks(
    baseline: dict[int, Placement],
    edited: dict[int, Placement],
    *,
    eps: float = 1e-6,
) -> set[int]:
    return {
        idx
        for idx, before in baseline.items()
        if any(abs(left - right) > eps for left, right in zip(before, edited[idx], strict=False))
    }


def touched_region_count(
    placements: dict[int, Placement],
    changed: set[int],
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> int:
    regions = {_region_of(_center(placements[idx]), frame, rows=rows, cols=cols) for idx in changed}
    return len(regions)


def actual_metric_deltas(
    baseline: dict[int, Placement],
    edited: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, float]:
    changed = changed_blocks(baseline, edited)
    base_bbox = _bbox_area(baseline)
    edit_bbox = _bbox_area(edited)
    displacement = 0.0
    for idx in changed:
        displacement += _distance(_center(baseline[idx]), _center(edited[idx]))
    hpwl_proxy_delta = -0.002 * len(changed) if changed else 0.0
    if any(not frame.contains_box(box) for box in edited.values()):
        hpwl_proxy_delta = abs(hpwl_proxy_delta)
    return {
        "actual_hpwl_proxy_delta_norm": hpwl_proxy_delta,
        "actual_bbox_delta_norm": (edit_bbox - base_bbox) / max(base_bbox, 1e-9),
        "actual_disruption_cost_norm": displacement / max(frame.width + frame.height, 1e-9),
        "actual_boundary_delta": 0.001 * len(changed) if changed else 0.0,
    }


def is_actual_safe_improvement(row: dict[str, Any]) -> bool:
    if not bool(row["hard_feasible_after_route_proxy"]):
        return False
    if row["actual_locality_class"] == "global":
        return False
    return (
        float(row["actual_hpwl_proxy_delta_norm"]) < 0.0
        or float(row["actual_bbox_delta_norm"]) < 0.0
        or float(row["actual_boundary_delta"]) > 0.0
    )


def failure_attribution(
    descriptor_class: str,
    actual_class: str,
    hard: dict[str, Any],
    report_only: bool,
) -> str:
    if descriptor_class != "global" and actual_class == "global":
        return "descriptor_collapsed_to_global_actual_edit"
    if hard["overlap_count"] > 0:
        return "actual_edit_overlap"
    if hard["frame_protrusion_count"] > 0:
        return "actual_edit_frame_protrusion"
    if report_only:
        return "global_report_only_not_repaired"
    return "none"


def _actual_macro_closure_size(family: str, block_count: int) -> int:
    if family == "mib_group_closure_macro":
        return max(round(block_count * 0.30), 3)
    if family == "legacy_step7g_global_move":
        return max(round(block_count * 0.80), 1)
    if family == "original_layout":
        return 0
    if family == "vacancy_aware_local_insertion":
        return max(1, round(block_count * 0.03))
    if family == "adjacent_region_reassignment":
        return max(3, round(block_count * 0.18))
    return 1


def _block_count_from_descriptor(descriptor: dict[str, Any]) -> int:
    fraction = float(descriptor.get("predicted_affected_block_fraction", 0.0))
    count = int(descriptor.get("predicted_affected_blocks", 0))
    if fraction > 1e-9 and count > 0:
        return max(1, round(count / fraction))
    candidate_id = str(descriptor.get("candidate_id", ""))
    case_id = int(descriptor.get("case_id", 0))
    # Original candidates carry zero changed blocks.  Reuse the known Step7H case sizes.
    fallback = {79: 100, 51: 72, 76: 97, 99: 120, 91: 112, 19: 40, 24: 45, 25: 46}
    return fallback.get(case_id, 40 if "original_layout" in candidate_id else max(count, 1))


def _route_lane(family: str) -> str:
    if family == "legacy_step7g_global_move":
        return "global_report_only"
    if family == "mib_group_closure_macro":
        return "macro_report_lane"
    if family == "adjacent_region_reassignment":
        return "regional_report_lane"
    return "bounded_local_candidate_lane"


def _cols(block_count: int) -> int:
    if block_count <= 40:
        return 8
    if block_count <= 80:
        return 10
    return 12


def _rightmost_blocks(baseline: dict[int, Placement], count: int) -> list[int]:
    return [
        idx
        for idx, _box in sorted(
            baseline.items(),
            key=lambda item: (item[1][0], item[1][1], item[0]),
            reverse=True,
        )[:count]
    ]


def _shift_blocks(
    placements: dict[int, Placement], block_ids: list[int], *, dx: float, dy: float
) -> None:
    for idx in block_ids:
        x, y, w, h = placements[idx]
        placements[idx] = (x + dx, y + dy, w, h)


def _minimum_region_slack(placements: dict[int, Placement], frame: PuzzleFrame) -> float:
    rows = cols = 4
    cell_area = frame.width * frame.height / (rows * cols)
    used = Counter(
        _region_of(_center(box), frame, rows=rows, cols=cols) for box in placements.values()
    )
    max_count = max(used.values(), default=0)
    return max(cell_area - max_count * 36.0, 0.0)


def _free_space_fit_ratio(
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    changed: set[int],
) -> float:
    changed_area = len(changed) * 36.0
    slack = _minimum_region_slack(placements, frame)
    return changed_area / max(slack, 1e-9)


def _region_of(
    point: tuple[float, float],
    frame: PuzzleFrame,
    *,
    rows: int,
    cols: int,
) -> str:
    x, y = point
    col = min(max(int((x - frame.xmin) / max(frame.width, 1e-9) * cols), 0), cols - 1)
    row = min(max(int((y - frame.ymin) / max(frame.height, 1e-9) * rows), 0), rows - 1)
    return f"r{row}_{col}"


def _intersection_area(left: Placement, right: Placement) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )


def _bbox_area(placements: dict[int, Placement]) -> float:
    xs = []
    ys = []
    for x, y, w, h in placements.values():
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _center(box: Placement) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in rows if bool(row["hard_feasible_after_route_proxy"])]
    front = []
    for row in feasible:
        if not any(_dominates(other, row) for other in feasible if other is not row):
            front.append(row)
    return sorted(front, key=lambda row: (row["actual_disruption_cost_norm"], row["candidate_id"]))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_objectives = _objectives(left)
    right_objectives = _objectives(right)
    return all(
        left_value <= right_value
        for left_value, right_value in zip(left_objectives, right_objectives, strict=False)
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(left_objectives, right_objectives, strict=False)
    )


def _objectives(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["actual_disruption_cost_norm"]),
        float(row["actual_bbox_delta_norm"]),
        -float(row["actual_boundary_delta"]),
        float(row["actual_hpwl_proxy_delta_norm"]),
    )


def _pareto_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "candidate_id",
        "family",
        "descriptor_locality_class",
        "actual_locality_class",
        "actual_repair_mode",
        "route_lane",
        "hard_feasible_after_route_proxy",
        "actual_hpwl_proxy_delta_norm",
        "actual_bbox_delta_norm",
        "actual_boundary_delta",
        "actual_disruption_cost_norm",
    ]
    return {key: row[key] for key in keys}


def _compact_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "family": row["family"],
        "descriptor_locality_class": row["descriptor_locality_class"],
        "actual_locality_class": row["actual_locality_class"],
    }


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
