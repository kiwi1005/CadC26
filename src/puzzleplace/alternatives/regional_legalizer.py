from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.alternatives.pareto_slack_fit_edits import DEFAULT_EPSILON, objective_vector
from puzzleplace.alternatives.real_placement_edits import (
    frame_from_baseline,
    free_space_fit_ratio,
    metric_delta_report,
    minimum_region_slack,
    official_like_hard_summary,
    placements_from_case,
    route_appropriate_real_repair_proxy,
    touched_region_count,
)
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.region_topology import (
    block_area,
    box_center,
    build_region_grid,
    point_region,
    region_center,
)
from puzzleplace.diagnostics.spatial_locality import macro_closure_blocks
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

GRID_ROWS = 4
GRID_COLS = 4
STEP7I_INFEASIBLE_AFTER_REGIONAL_EDIT_BASELINE = 32


@dataclass(frozen=True)
class RegionalLegalizerCandidate:
    case_id: int
    candidate_id: str
    source_candidate_id: str
    assignment_type: str
    source_actual_locality_class: str
    descriptor_locality_class: str
    descriptor_repair_mode: str
    legalizer_strategy: str
    legalizer_status: str
    legalizer_notes: tuple[str, ...]
    case: FloorSetCase
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    changed_blocks: tuple[int, ...]
    source_changed_blocks: tuple[int, ...]
    macro_closure_blocks: tuple[int, ...]
    target_region: str
    assignment_cost: dict[str, float]
    construction_ms: float


def build_regional_legalizer_candidates(
    step7i_rows: list[dict[str, Any]], cases_by_id: dict[int, FloorSetCase]
) -> tuple[list[RegionalLegalizerCandidate], list[dict[str, Any]]]:
    candidates: list[RegionalLegalizerCandidate] = []
    attempts: list[dict[str, Any]] = []
    for row in step7i_rows:
        case_id = int(row["case_id"])
        if case_id not in cases_by_id:
            continue
        case = cases_by_id[case_id]
        baseline = placements_from_case(case)
        frame = frame_from_baseline(baseline)
        candidate, attempt = legalize_step7i_candidate_row(row, case, baseline, frame)
        candidates.append(candidate)
        attempts.append(attempt)
    return candidates, attempts


def legalize_step7i_candidate_row(
    row: dict[str, Any],
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
) -> tuple[RegionalLegalizerCandidate, dict[str, Any]]:
    t0 = time.perf_counter()
    assignment_type = str(row["assignment_type"])
    source_class = str(row["actual_locality_class"])
    source_changed = tuple(int(idx) for idx in row.get("real_block_ids_changed", []))
    assignment_cost = {key: float(value) for key, value in row.get("assignment_cost", {}).items()}
    target_region = str(row.get("target_region", "unknown"))

    notes: tuple[str, ...]
    if assignment_type == "original_layout":
        edited = dict(baseline)
        status = "original_anchor"
        strategy = "original_anchor"
        notes = ("original-inclusive baseline",)
    elif assignment_type == "legacy_step7g_global_report" or source_class == "global":
        edited = _reconstruct_raw_or_baseline(row, baseline)
        status = "global_report_only"
        strategy = "global_report_only"
        notes = ("global source candidate kept report-only; no regional legalization",)
    elif source_class == "macro" or int(row.get("macro_closure_size", 0)) > len(source_changed):
        edited = dict(baseline)
        status = "macro_closure_requires_supernode"
        strategy = "supernode_closure_probe"
        notes = ("macro/MIB/group closure requires supernode expansion before placement",)
    else:
        strategy = "slot_assignment_matching"
        edited, status, notes = _slot_assignment_matching(row, case, baseline, frame)

    changed = tuple(sorted(_changed_blocks(baseline, edited)))
    closure = tuple(sorted(macro_closure_blocks(case, set(changed)))) if changed else ()
    candidate_id = f"{row['candidate_id']}:legalized:{strategy}"
    candidate = RegionalLegalizerCandidate(
        case_id=int(row["case_id"]),
        candidate_id=candidate_id,
        source_candidate_id=str(row["candidate_id"]),
        assignment_type=assignment_type,
        source_actual_locality_class=source_class,
        descriptor_locality_class=str(row.get("descriptor_locality_class", source_class)),
        descriptor_repair_mode=str(row.get("descriptor_repair_mode", "region_repair_or_planner")),
        legalizer_strategy=strategy,
        legalizer_status=status,
        legalizer_notes=tuple(notes),
        case=case,
        baseline=baseline,
        edited=edited,
        frame=frame,
        changed_blocks=changed,
        source_changed_blocks=source_changed,
        macro_closure_blocks=closure,
        target_region=target_region,
        assignment_cost=assignment_cost,
        construction_ms=(time.perf_counter() - t0) * 1000.0,
    )
    attempt = {
        "case_id": row["case_id"],
        "source_candidate_id": row["candidate_id"],
        "legalizer_candidate_id": candidate_id,
        "source_actual_locality_class": source_class,
        "assignment_type": assignment_type,
        "legalizer_strategy": strategy,
        "legalizer_status": status,
        "source_changed_block_count": len(source_changed),
        "changed_block_count_after_legalizer": len(changed),
        "no_op_after_legalizer": len(changed) == 0 and assignment_type != "original_layout",
        "target_region": target_region,
        "notes": list(notes),
        "runtime_ms": candidate.construction_ms,
    }
    return candidate, attempt


def evaluate_regional_legalizer_candidate(candidate: RegionalLegalizerCandidate) -> dict[str, Any]:
    hard_before = official_like_hard_summary(candidate.case, candidate.edited, candidate.frame)
    changed = set(candidate.changed_blocks)
    regions = touched_region_count(candidate.edited, changed, candidate.frame)
    fit_ratio = free_space_fit_ratio(candidate.edited, changed, candidate.frame)
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.case.block_count,
        changed_block_count=len(changed),
        touched_region_count=regions,
        macro_closure_size=len(candidate.macro_closure_blocks),
        min_region_slack=minimum_region_slack(candidate.edited, candidate.frame),
        free_space_fit_ratio=fit_ratio,
        hard_summary={"hard_feasible": hard_before["official_like_hard_feasible"]},
    )
    actual_class = str(prediction["predicted_locality_class"])
    repair = route_appropriate_real_repair_proxy(candidate, actual_class, hard_before)  # type: ignore[arg-type]
    metrics = metric_delta_report(candidate.case, candidate.baseline, repair["after_route"])
    row: dict[str, Any] = {
        "case_id": candidate.case_id,
        "candidate_id": candidate.candidate_id,
        "source_candidate_id": candidate.source_candidate_id,
        "strategy": candidate.assignment_type,
        "assignment_type": candidate.assignment_type,
        "source_actual_locality_class": candidate.source_actual_locality_class,
        "descriptor_locality_class": candidate.descriptor_locality_class,
        "descriptor_repair_mode": candidate.descriptor_repair_mode,
        "actual_locality_class": actual_class,
        "actual_repair_mode": prediction["predicted_repair_mode"],
        "route_lane": repair["route_lane"],
        "report_only": repair["report_only"],
        "legalizer_strategy": candidate.legalizer_strategy,
        "legalizer_status": candidate.legalizer_status,
        "legalizer_notes": list(candidate.legalizer_notes),
        "real_block_ids_changed": list(candidate.changed_blocks),
        "source_block_ids_changed": list(candidate.source_changed_blocks),
        "macro_closure_block_ids": list(candidate.macro_closure_blocks),
        "changed_block_count": len(changed),
        "source_changed_block_count": len(candidate.source_changed_blocks),
        "changed_block_fraction": len(changed) / max(candidate.case.block_count, 1),
        "affected_region_count": regions,
        "macro_closure_size": len(candidate.macro_closure_blocks),
        "free_space_fit_ratio": fit_ratio,
        "target_region": candidate.target_region,
        "assignment_cost": candidate.assignment_cost,
        "assignment_total_cost": sum(candidate.assignment_cost.values()),
        **{f"before_repair_{key}": value for key, value in hard_before.items()},
        **{f"after_route_{key}": value for key, value in repair["after_hard"].items()},
        **metrics,
    }
    row["runtime_proxy_ms"] = candidate.construction_ms + float(metrics["metric_eval_ms"])
    row["no_op_after_legalizer"] = (
        len(changed) == 0 and candidate.assignment_type != "original_layout"
    )
    row["no_op"] = row["no_op_after_legalizer"]
    row["official_like_cost_improving"] = (
        bool(row["after_route_official_like_hard_feasible"])
        and float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON
    )
    row["hpwl_improving"] = float(row["hpwl_delta"]) < -DEFAULT_EPSILON
    row["safe_improvement"] = is_legalizer_safe_improvement(row)
    row["objective_vector"] = objective_vector(row)
    row["failure_attribution"] = failure_attribution(row)
    return row


def evaluate_regional_legalizer_candidates(
    candidates: list[RegionalLegalizerCandidate],
) -> list[dict[str, Any]]:
    return [evaluate_regional_legalizer_candidate(candidate) for candidate in candidates]


def feasibility_report(
    rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    attempted_regional = [row for row in rows if row["source_actual_locality_class"] == "regional"]
    attempted_macro = [row for row in rows if row["source_actual_locality_class"] == "macro"]
    hard_feasible = [row for row in rows if row["after_route_official_like_hard_feasible"]]
    regional_improving = [
        row
        for row in rows
        if row["source_actual_locality_class"] == "regional" and row["official_like_cost_improving"]
    ]
    macro_improving = [
        row
        for row in rows
        if row["source_actual_locality_class"] == "macro" and row["official_like_cost_improving"]
    ]
    source_infeasible = sum(
        int(row.get("failure_attribution") == "infeasible_after_regional_edit")
        for row in source_rows
    )
    after_infeasible = sum(
        int(row["failure_attribution"] == "infeasible_after_regional_edit") for row in rows
    )
    per_case_runtime: dict[str, float] = defaultdict(float)
    for row in rows:
        per_case_runtime[str(row["case_id"])] += float(row["runtime_proxy_ms"])
    return {
        "source_step7i_candidate_count": len(source_rows),
        "attempted_regional_candidate_count": len(attempted_regional),
        "attempted_macro_candidate_count": len(attempted_macro),
        "legalizer_attempt_count_by_strategy": dict(
            Counter(str(row["legalizer_strategy"]) for row in rows)
        ),
        "route_count_before_legalizer": dict(
            Counter(str(row["actual_locality_class"]) for row in source_rows)
        ),
        "route_count_after_legalizer": dict(
            Counter(str(row["actual_locality_class"]) for row in rows)
        ),
        "hard_feasible_count_after_legalizer": len(hard_feasible),
        "hard_feasible_rate_after_legalizer": _rate(len(hard_feasible), len(rows)),
        "non_original_non_noop_hard_feasible_count_after_legalizer": sum(
            int(
                row["assignment_type"] != "original_layout"
                and not row["no_op_after_legalizer"]
                and bool(row["after_route_official_like_hard_feasible"])
            )
            for row in rows
        ),
        "infeasible_after_regional_edit_count": after_infeasible,
        "step7i_infeasible_after_regional_edit_baseline": source_infeasible,
        "overlap_resolved_count": sum(
            int(
                _source_overlap_for(row, source_rows) > 0
                and int(row["after_route_overlap_violation_count"]) == 0
            )
            for row in rows
        ),
        "no_feasible_slot_count": sum(
            int(row["failure_attribution"] == "no_feasible_slot") for row in rows
        ),
        "region_capacity_failure_count": sum(
            int(row["failure_attribution"] == "region_capacity_failure") for row in rows
        ),
        "fixed_preplaced_conflict_count": sum(
            int(row["failure_attribution"] == "fixed_preplaced_conflict") for row in rows
        ),
        "MIB_group_closure_conflict_count": sum(
            int(row["failure_attribution"] == "MIB_group_closure_conflict") for row in rows
        ),
        "macro_closure_requires_supernode_count": sum(
            int(row["failure_attribution"] == "macro_closure_requires_supernode") for row in rows
        ),
        "no_op_after_legalizer_count": sum(int(row["no_op_after_legalizer"]) for row in rows),
        "official_like_improving_candidate_count": sum(
            int(row["official_like_cost_improving"]) for row in rows
        ),
        "regional_official_like_improving_count": len(regional_improving),
        "macro_official_like_improving_count": len(macro_improving),
        "official_like_improvement_density": _rate(
            sum(int(row["official_like_cost_improving"]) for row in hard_feasible),
            len(hard_feasible),
        ),
        "runtime_proxy_per_case": dict(sorted(per_case_runtime.items())),
    }


def failure_attribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": dict(Counter(str(row["failure_attribution"]) for row in rows)),
        "rows": [
            _compact(row) | {"failure_attribution": row["failure_attribution"]}
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def local_starvation_recovery_report(
    rows: list[dict[str, Any]], step7i_starvation: dict[str, Any]
) -> dict[str, Any]:
    no_official_cases = [
        int(case) for case in step7i_starvation.get("no_official_like_local_candidate_cases", [])
    ]
    no_retained_cases = [
        int(case) for case in step7i_starvation.get("no_preselection_retained_cases", [])
    ]
    improving_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["official_like_cost_improving"]:
            improving_by_case[int(row["case_id"])].append(row)
    recovered_no_official = [case for case in no_official_cases if case in improving_by_case]
    recovered_no_retained = [case for case in no_retained_cases if case in improving_by_case]
    return {
        "local_starvation_case_recovery_count": len(
            set(recovered_no_official) | set(recovered_no_retained)
        ),
        "cases_recovered_from_no_official_like_local_candidate": recovered_no_official,
        "cases_recovered_from_no_preselection_retained": recovered_no_retained,
        "official_like_improving_by_case": {
            str(case): [_compact(row) for row in group]
            for case, group in sorted(improving_by_case.items())
        },
    }


def decision_for_step7i_r(feasibility: dict[str, Any], starvation: dict[str, Any]) -> str:
    feasible_after = int(feasibility["hard_feasible_count_after_legalizer"])
    feasible_rate = float(feasibility["hard_feasible_rate_after_legalizer"])
    improved = int(feasibility["official_like_improving_candidate_count"])
    recovered = int(starvation["local_starvation_case_recovery_count"])
    no_op = int(feasibility["no_op_after_legalizer_count"])
    attempted = int(feasibility["attempted_regional_candidate_count"]) + int(
        feasibility["attempted_macro_candidate_count"]
    )
    if improved > 0 and recovered > 0 and feasible_rate >= 0.50:
        return "promote_regional_legalizer_to_step7c_hybrid"
    if int(feasibility["macro_closure_requires_supernode_count"]) >= max(1, attempted // 2):
        return "pivot_to_macro_closure_generator"
    if feasible_after > 8 and no_op < attempted:
        return "refine_region_assignment_costs" if improved == 0 else "refine_regional_legalizer"
    if feasible_after > 8:
        return "refine_regional_legalizer"
    return "inconclusive_due_to_legalizer_quality"


def is_legalizer_safe_improvement(row: dict[str, Any]) -> bool:
    return (
        bool(row["after_route_official_like_hard_feasible"])
        and row["actual_locality_class"] != "global"
        and (
            row["official_like_cost_improving"]
            or float(row["hpwl_delta"]) < -DEFAULT_EPSILON
            or float(row["bbox_area_delta"]) < -DEFAULT_EPSILON
        )
    )


def failure_attribution(row: dict[str, Any]) -> str:
    if row["assignment_type"] == "original_layout":
        return "none"
    if row["legalizer_status"] == "global_report_only" or row["actual_locality_class"] == "global":
        return "route_global"
    if row["legalizer_status"] == "macro_closure_requires_supernode":
        return "macro_closure_requires_supernode"
    if row["legalizer_status"] == "fixed_preplaced_conflict":
        return "fixed_preplaced_conflict"
    if row["legalizer_status"] == "region_capacity_failure":
        return "region_capacity_failure"
    if row["legalizer_status"] == "no_feasible_slot":
        return "no_feasible_slot"
    if row["no_op_after_legalizer"]:
        return "no_op_after_legalizer"
    if not bool(row["after_route_official_like_hard_feasible"]):
        return "infeasible_after_regional_edit"
    if row["official_like_cost_improving"]:
        return "none"
    return "metric_non_improving"


def _slot_assignment_matching(
    row: dict[str, Any],
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
) -> tuple[dict[int, Placement], str, tuple[str, ...]]:
    changed = [int(idx) for idx in row.get("real_block_ids_changed", [])]
    changed = [idx for idx in changed if idx in baseline and _is_movable(case, idx)]
    if not changed:
        return dict(baseline), "no_feasible_slot", ("no movable regional blocks",)
    source_regions = [str(rid) for rid in row.get("source_regions", [])]
    target_region = str(row.get("target_region", "r0_0"))
    window_regions = _window_region_ids(source_regions + [target_region])
    cells = {
        str(cell["region_id"]): cell
        for cell in build_region_grid(frame, rows=GRID_ROWS, cols=GRID_COLS)
    }
    window_cells = [cells[rid] for rid in sorted(window_regions) if rid in cells]
    if not window_cells:
        return dict(baseline), "no_compatible_region", ("empty regional window",)
    window = _window_bbox(window_cells)
    capacity = (window[2] - window[0]) * (window[3] - window[1])
    affected = sorted(
        {
            idx
            for idx, box in baseline.items()
            if _is_movable(case, idx)
            and point_region(*box_center(box), frame, rows=GRID_ROWS, cols=GRID_COLS)
            in window_regions
        }
        | set(changed)
    )
    total_area = sum(block_area(baseline[idx]) for idx in affected)
    if total_area > capacity * 1.05:
        return dict(baseline), "region_capacity_failure", ("regional window area is too small",)
    static = {idx: box for idx, box in baseline.items() if idx not in set(changed)}
    edited = dict(baseline)
    assigned: dict[int, Placement] = {}
    moved_count = 0
    for idx in sorted(changed, key=lambda bid: (-block_area(baseline[bid]), bid)):
        slot = _best_slot_for_block(idx, baseline, row, frame, window, static, assigned)
        if slot is None:
            slot = baseline[idx]
        assigned[idx] = slot
        edited[idx] = slot
        if _placement_distance(slot, baseline[idx]) > 1e-9:
            moved_count += 1
    hard = official_like_hard_summary(case, edited, frame)
    if not hard["official_like_hard_feasible"]:
        # Fall back to a deterministic no-op legalized state rather than leaking raw infeasibility.
        return (
            dict(baseline),
            "no_feasible_slot",
            ("slot matching could not legalize without overlap",),
        )
    if moved_count == 0:
        return edited, "no_feasible_slot", ("only original slots were feasible",)
    return edited, "legalized_slot_assignment", (f"slot-assigned {moved_count} moved blocks",)


def _best_slot_for_block(
    idx: int,
    baseline: dict[int, Placement],
    row: dict[str, Any],
    frame: PuzzleFrame,
    window: tuple[float, float, float, float],
    static: dict[int, Placement],
    assigned: dict[int, Placement],
) -> Placement | None:
    x, y, w, h = baseline[idx]
    target_region = str(row.get("target_region", "r0_0"))
    cells = {
        str(cell["region_id"]): cell
        for cell in build_region_grid(frame, rows=GRID_ROWS, cols=GRID_COLS)
    }
    target_center = (
        region_center(cells[target_region]) if target_region in cells else box_center(baseline[idx])
    )
    candidates = _candidate_slots((w, h), window, target_center) + [baseline[idx]]
    blockers = list(static.values()) + list(assigned.values())
    feasible = [
        slot
        for slot in candidates
        if _inside_frame(slot, frame) and not any(_overlap(slot, other) for other in blockers)
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda box: (
            _distance(box_center(box), target_center),
            _placement_distance(box, baseline[idx]),
        ),
    )


def _candidate_slots(
    size: tuple[float, float],
    window: tuple[float, float, float, float],
    target: tuple[float, float],
) -> list[Placement]:
    w, h = size
    xmin, ymin, xmax, ymax = window
    xs = _linspace(xmin, max(xmin, xmax - w), 5)
    ys = _linspace(ymin, max(ymin, ymax - h), 5)
    raw = [(x, y, w, h) for x in xs for y in ys]
    raw.append((target[0] - w / 2.0, target[1] - h / 2.0, w, h))
    unique: dict[tuple[float, float, float, float], Placement] = {}
    for slot in raw:
        clamped = (
            min(max(slot[0], xmin), max(xmin, xmax - w)),
            min(max(slot[1], ymin), max(ymin, ymax - h)),
            w,
            h,
        )
        unique[(round(clamped[0], 8), round(clamped[1], 8), round(w, 8), round(h, 8))] = clamped
    return list(unique.values())


def _window_region_ids(region_ids: list[str]) -> set[str]:
    out: set[str] = set()
    for rid in region_ids:
        if not rid.startswith("r") or "_" not in rid:
            continue
        raw = rid.removeprefix("r")
        row, col = [int(v) for v in raw.split("_", 1)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                    out.add(f"r{rr}_{cc}")
    return out


def _window_bbox(cells: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(float(cell["xmin"]) for cell in cells),
        min(float(cell["ymin"]) for cell in cells),
        max(float(cell["xmax"]) for cell in cells),
        max(float(cell["ymax"]) for cell in cells),
    )


def _reconstruct_raw_or_baseline(
    row: dict[str, Any], baseline: dict[int, Placement]
) -> dict[int, Placement]:
    del row
    return dict(baseline)


def _source_overlap_for(row: dict[str, Any], source_rows: list[dict[str, Any]]) -> int:
    source_id = row["source_candidate_id"]
    for source in source_rows:
        if source["candidate_id"] == source_id:
            return int(source.get("after_route_overlap_violation_count", 0))
    return 0


def _is_movable(case: FloorSetCase, idx: int) -> bool:
    return not bool(case.constraints[idx, ConstraintColumns.FIXED].item()) and not bool(
        case.constraints[idx, ConstraintColumns.PREPLACED].item()
    )


def _changed_blocks(baseline: dict[int, Placement], edited: dict[int, Placement]) -> set[int]:
    return {
        idx
        for idx, before in baseline.items()
        if idx in edited
        and any(abs(a - b) > 1e-9 for a, b in zip(before, edited[idx], strict=False))
    }


def _inside_frame(box: Placement, frame: PuzzleFrame) -> bool:
    return frame.contains_box(box)


def _overlap(left: Placement, right: Placement) -> bool:
    return not (
        left[0] + left[2] <= right[0] + 1e-9
        or right[0] + right[2] <= left[0] + 1e-9
        or left[1] + left[3] <= right[1] + 1e-9
        or right[1] + right[3] <= left[1] + 1e-9
    )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1 or abs(stop - start) <= 1e-12:
        return [start]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


def _placement_distance(left: Placement, right: Placement) -> float:
    return _distance(box_center(left), box_center(right))


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "source_candidate_id": row["source_candidate_id"],
        "assignment_type": row["assignment_type"],
        "source_actual_locality_class": row["source_actual_locality_class"],
        "actual_locality_class": row["actual_locality_class"],
        "legalizer_strategy": row["legalizer_strategy"],
        "legalizer_status": row["legalizer_status"],
        "after_route_official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
        "official_like_cost_delta": row["official_like_cost_delta"],
        "hpwl_delta": row["hpwl_delta"],
        "bbox_area_delta": row["bbox_area_delta"],
        "soft_constraint_delta": row["mib_group_boundary_soft_delta"]["total_delta"],
        "changed_block_count": row["changed_block_count"],
        "no_op_after_legalizer": row["no_op_after_legalizer"],
    }


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _distribution(values: Any) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "median": median(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }
