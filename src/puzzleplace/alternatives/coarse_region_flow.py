from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from puzzleplace.alternatives.locality_routing import predict_move_locality
from puzzleplace.alternatives.pareto_slack_fit_edits import (
    DEFAULT_EPSILON,
    objective_vector,
)
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
    expected_block_centroid,
    net_community_clusters,
    point_region,
    region_center,
    region_grid_distance,
    region_occupancy,
)
from puzzleplace.diagnostics.spatial_locality import macro_closure_blocks
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

GRID_ROWS = 4
GRID_COLS = 4
MAX_ASSIGNMENTS_PER_CASE = 5


@dataclass(frozen=True)
class CoarseRegionAssignmentCandidate:
    case_id: int
    candidate_id: str
    assignment_type: str
    descriptor_locality_class: str
    descriptor_repair_mode: str
    case: FloorSetCase
    baseline: dict[int, Placement]
    edited: dict[int, Placement]
    frame: PuzzleFrame
    changed_blocks: tuple[int, ...]
    macro_closure_blocks: tuple[int, ...]
    source_regions: tuple[str, ...]
    target_region: str
    assignment_cost: dict[str, float]
    construction_status: str
    construction_notes: tuple[str, ...]
    construction_ms: float


def build_region_maps(case_ids: list[int], cases_by_id: dict[int, FloorSetCase]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        case = cases_by_id[case_id]
        placements = placements_from_case(case)
        frame = frame_from_baseline(placements)
        occ = region_occupancy(case, placements, frame, rows=GRID_ROWS, cols=GRID_COLS)
        cases.append(
            {
                "case_id": case_id,
                "grid": {"rows": GRID_ROWS, "cols": GRID_COLS},
                "regions": [_region_row(case, placements, row) for row in occ["regions"]],
                "summary": {
                    "region_count": len(occ["regions"]),
                    "max_utilization": occ["max_utilization"],
                    "min_utilization": occ["min_utilization"],
                    "utilization_spread": occ["utilization_spread"],
                    "overflow_region_count": occ["overflow_region_count"],
                    "total_unused_capacity": sum(
                        float(r["unused_capacity"]) for r in occ["regions"]
                    ),
                    "total_overflow_area": sum(float(r["overflow_area"]) for r in occ["regions"]),
                },
            }
        )
    return {
        "grid_policy": "deterministic 4x4 coarse frame grid",
        "real_case_count": len(cases),
        "coarse_region_count_by_case": {str(row["case_id"]): len(row["regions"]) for row in cases},
        "region_capacity_slack_summary": {str(row["case_id"]): row["summary"] for row in cases},
        "cases": cases,
    }


def build_flow_assignment_candidates(
    case_ids: list[int], cases_by_id: dict[int, FloorSetCase]
) -> list[CoarseRegionAssignmentCandidate]:
    out: list[CoarseRegionAssignmentCandidate] = []
    for case_id in sorted(case_ids):
        case = cases_by_id[case_id]
        baseline = placements_from_case(case)
        frame = frame_from_baseline(baseline)
        out.append(_original_candidate(case_id, case, baseline, frame))
        assignment_rows = coarse_assignment_rows(case, baseline, frame)
        for variant_index, assignment in enumerate(assignment_rows[:MAX_ASSIGNMENTS_PER_CASE]):
            out.append(
                _candidate_from_assignment(
                    case_id, case, baseline, frame, assignment, variant_index
                )
            )
        out.append(_global_report_candidate(case_id, case, baseline, frame))
    return out


def coarse_assignment_rows(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> list[dict[str, Any]]:
    cells = build_region_grid(frame, rows=GRID_ROWS, cols=GRID_COLS)
    cell_by_id = {str(cell["region_id"]): cell for cell in cells}
    occupancy = region_occupancy(case, placements, frame, rows=GRID_ROWS, cols=GRID_COLS)
    region_by_id = {str(row["region_id"]): row for row in occupancy["regions"]}
    clusters = net_community_clusters(case)["clusters"]
    group_size = max(2, math.ceil(case.block_count * 0.15))
    seed_groups = _seed_groups(case, placements, clusters, group_size=group_size)
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seed_groups):
        source_regions = sorted(
            {
                point_region(*box_center(placements[idx]), frame, rows=GRID_ROWS, cols=GRID_COLS)
                for idx in seed["members"]
            }
        )
        source_region = source_regions[0] if source_regions else "r0_0"
        current_center = _group_center(placements, seed["members"])
        demand_center = _group_demand_center(case, placements, seed["members"])
        demand_region = point_region(*demand_center, frame, rows=GRID_ROWS, cols=GRID_COLS)
        for target_id, cell in cell_by_id.items():
            if target_id in source_regions and target_id == demand_region:
                continue
            cost = assignment_cost_components(
                case,
                placements,
                frame,
                seed["members"],
                source_region=source_region,
                target_region=target_id,
                target_cell=cell,
                region=row_with_default(region_by_id, target_id),
                current_center=current_center,
                demand_center=demand_center,
            )
            rows.append(
                {
                    "seed_index": seed_index,
                    "assignment_type": seed["assignment_type"],
                    "members": seed["members"],
                    "source_regions": source_regions,
                    "target_region": target_id,
                    "target_center": list(region_center(cell)),
                    "demand_center": list(demand_center),
                    "assignment_cost": cost,
                    "total_cost": sum(cost.values()),
                }
            )
    return _select_assignments(rows)


def assignment_cost_components(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    members: list[int],
    *,
    source_region: str,
    target_region: str,
    target_cell: dict[str, Any],
    region: dict[str, Any],
    current_center: tuple[float, float],
    demand_center: tuple[float, float],
) -> dict[str, float]:
    target_center = region_center(target_cell)
    group_area = sum(block_area(placements[idx]) for idx in members)
    slack = float(region.get("unused_capacity", 0.0))
    hpwl_demand = _distance(target_center, demand_center) / max(
        math.hypot(frame.width, frame.height), 1e-9
    )
    slack_shortfall = max(group_area - slack, 0.0) / max(group_area, 1e-9)
    bbox_pressure = _bbox_pressure_cost(frame, current_center, target_center)
    fixed_preplaced = sum(
        int(bool(case.constraints[idx, ConstraintColumns.FIXED].item()))
        + int(bool(case.constraints[idx, ConstraintColumns.PREPLACED].item()))
        for idx in members
    ) / max(len(members), 1)
    closure = macro_closure_blocks(case, set(members))
    mib_risk = max(len(closure) - len(members), 0) / max(case.block_count, 1)
    boundary_members = [
        idx for idx in members if bool(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
    ]
    boundary_cost = 0.0
    if boundary_members:
        boundary_cost = region_grid_distance(source_region, target_region) / max(
            GRID_ROWS + GRID_COLS - 2, 1
        )
    return {
        "hpwl_community_demand": hpwl_demand,
        "region_slack_capacity": slack_shortfall,
        "bbox_pressure": bbox_pressure,
        "fixed_preplaced_incompatibility": fixed_preplaced,
        "mib_group_closure_risk": mib_risk,
        "boundary_ownership_compatibility": boundary_cost,
    }


def evaluate_flow_assignment_candidate(
    candidate: CoarseRegionAssignmentCandidate,
) -> dict[str, Any]:
    hard_before = official_like_hard_summary(candidate.case, candidate.edited, candidate.frame)
    changed = set(candidate.changed_blocks)
    regions = touched_region_count(candidate.edited, changed, candidate.frame)
    raw_fit = free_space_fit_ratio(candidate.edited, changed, candidate.frame)
    prediction = predict_move_locality(
        case_id=candidate.case_id,
        block_count=candidate.case.block_count,
        changed_block_count=len(changed),
        touched_region_count=regions,
        macro_closure_size=len(candidate.macro_closure_blocks),
        min_region_slack=minimum_region_slack(candidate.edited, candidate.frame),
        free_space_fit_ratio=raw_fit,
        hard_summary={"hard_feasible": hard_before["official_like_hard_feasible"]},
    )
    actual_class = str(prediction["predicted_locality_class"])
    repair = route_appropriate_real_repair_proxy(candidate, actual_class, hard_before)  # type: ignore[arg-type]
    metrics = metric_delta_report(candidate.case, candidate.baseline, repair["after_route"])
    row: dict[str, Any] = {
        "case_id": candidate.case_id,
        "candidate_id": candidate.candidate_id,
        "strategy": candidate.assignment_type,
        "assignment_type": candidate.assignment_type,
        "descriptor_locality_class": candidate.descriptor_locality_class,
        "descriptor_repair_mode": candidate.descriptor_repair_mode,
        "actual_locality_class": actual_class,
        "actual_repair_mode": prediction["predicted_repair_mode"],
        "route_lane": repair["route_lane"],
        "report_only": repair["report_only"],
        "construction_status": candidate.construction_status,
        "construction_notes": list(candidate.construction_notes),
        "real_block_ids_changed": list(candidate.changed_blocks),
        "macro_closure_block_ids": list(candidate.macro_closure_blocks),
        "changed_block_count": len(changed),
        "changed_block_fraction": len(changed) / max(candidate.case.block_count, 1),
        "affected_region_count": regions,
        "macro_closure_size": len(candidate.macro_closure_blocks),
        "free_space_fit_ratio": raw_fit,
        "source_regions": list(candidate.source_regions),
        "target_region": candidate.target_region,
        "assignment_cost": candidate.assignment_cost,
        "assignment_total_cost": sum(candidate.assignment_cost.values()),
        **{f"before_repair_{key}": value for key, value in hard_before.items()},
        **{f"after_route_{key}": value for key, value in repair["after_hard"].items()},
        **metrics,
    }
    row["runtime_proxy_ms"] = candidate.construction_ms + float(metrics["metric_eval_ms"])
    row["no_op"] = len(changed) == 0 and candidate.assignment_type != "original_layout"
    row["official_like_cost_improving"] = (
        bool(row["after_route_official_like_hard_feasible"])
        and float(row["official_like_cost_delta"]) < -DEFAULT_EPSILON
    )
    row["hpwl_improving"] = float(row["hpwl_delta"]) < -DEFAULT_EPSILON
    row["safe_improvement"] = is_flow_safe_improvement(row)
    row["objective_vector"] = objective_vector(row)
    row["failure_attribution"] = failure_attribution(row)
    return row


def evaluate_flow_assignment_candidates(
    candidates: list[CoarseRegionAssignmentCandidate],
) -> list[dict[str, Any]]:
    return [evaluate_flow_assignment_candidate(candidate) for candidate in candidates]


def feasibility_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["after_route_official_like_hard_feasible"]]
    regional = [row for row in rows if row["actual_locality_class"] == "regional"]
    non_noop_regional = [row for row in regional if not row["no_op"]]
    promising = [
        row
        for row in rows
        if row["actual_locality_class"] in {"regional", "macro"}
        and not row["no_op"]
        and float(row["assignment_cost"].get("fixed_preplaced_incompatibility", 0.0))
        <= DEFAULT_EPSILON
        and row["failure_attribution"] != "route_global"
    ]
    per_case_runtime: dict[str, float] = defaultdict(float)
    for row in rows:
        per_case_runtime[str(row["case_id"])] += float(row["runtime_proxy_ms"])
    return {
        "real_case_count": len({int(row["case_id"]) for row in rows}),
        "assignment_candidate_count": len(rows),
        "candidate_count_by_assignment_type": dict(
            Counter(str(row["assignment_type"]) for row in rows)
        ),
        "route_count_by_class": dict(Counter(str(row["actual_locality_class"]) for row in rows)),
        "regional_candidate_count": len(regional),
        "global_candidate_count": sum(
            int(row["actual_locality_class"] == "global") for row in rows
        ),
        "non_global_candidate_rate": _rate(
            sum(int(row["actual_locality_class"] != "global") for row in rows), len(rows)
        ),
        "non_noop_regional_candidate_count": len(non_noop_regional),
        "regional_feasible_or_legalizer_promising_count": len(promising),
        "official_like_improving_candidate_count": sum(
            int(row["official_like_cost_improving"]) for row in rows
        ),
        "official_like_improvement_density": _rate(
            sum(int(row["official_like_cost_improving"]) for row in feasible), len(feasible)
        ),
        "official_like_hard_feasible_rate": _rate(len(feasible), len(rows)),
        "runtime_proxy_per_case": dict(sorted(per_case_runtime.items())),
    }


def failure_attribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": dict(Counter(str(row["failure_attribution"]) for row in rows)),
        "required_failure_counts": {
            key: sum(int(row["failure_attribution"] == key) for row in rows)
            for key in (
                "no_capacity",
                "no_compatible_region",
                "MIB_group_closure_conflict",
                "fixed_preplaced_conflict",
                "route_global",
                "no_op",
                "infeasible_after_regional_edit",
                "metric_non_improving",
            )
        },
        "rows": [
            _compact(row) | {"failure_attribution": row["failure_attribution"]}
            for row in rows
            if row["failure_attribution"] != "none"
        ],
    }


def local_starvation_recovery_report(
    rows: list[dict[str, Any]], iter0_trace: dict[str, Any]
) -> dict[str, Any]:
    starvation_rows = list(iter0_trace.get("rows", []))
    no_official = [
        row
        for row in starvation_rows
        if row.get("skip_reason") == "no_official_like_improving_local_candidate"
    ]
    no_retained = [
        row
        for row in starvation_rows
        if row.get("skip_reason") == "candidate_starvation_no_preselection_retained"
    ]
    improving_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    option_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["assignment_type"] in {"original_layout", "legacy_step7g_global_report"}:
            continue
        if row["actual_locality_class"] != "global" and not row["no_op"]:
            option_by_case[int(row["case_id"])].append(row)
        if row["official_like_cost_improving"]:
            improving_by_case[int(row["case_id"])].append(row)
    no_official_cases = sorted({int(row["case_id"]) for row in no_official})
    no_retained_cases = sorted({int(row["case_id"]) for row in no_retained})
    recovered_no_official = [case for case in no_official_cases if case in improving_by_case]
    recovered_no_retained = [case for case in no_retained_cases if case in improving_by_case]
    return {
        "local_starvation_events": len(starvation_rows),
        "no_official_like_local_candidate_events": len(no_official),
        "no_preselection_retained_events": len(no_retained),
        "no_official_like_local_candidate_cases": no_official_cases,
        "no_preselection_retained_cases": no_retained_cases,
        "cases_with_non_global_non_noop_region_options": sorted(option_by_case),
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


def assignment_cost_correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_names = sorted({key for row in rows for key in row.get("assignment_cost", {})})
    return {
        "assignment_cost_component_correlation_with_actual_delta": {
            name: _correlation_summary(
                [float(row["assignment_cost"].get(name, 0.0)) for row in rows],
                [float(row["official_like_cost_delta"]) for row in rows],
            )
            for name in component_names
        }
    }


def decision_for_step7i(
    feasibility: dict[str, Any], starvation: dict[str, Any], corr: dict[str, Any]
) -> str:
    improved = int(feasibility["official_like_improving_candidate_count"])
    regional = int(feasibility["regional_candidate_count"])
    promising = int(feasibility["regional_feasible_or_legalizer_promising_count"])
    recovered = int(starvation["local_starvation_case_recovery_count"])
    if regional == 0:
        return "revisit_region_probe_assumptions"
    if improved > 0 and recovered > 0 and float(feasibility["non_global_candidate_rate"]) >= 0.70:
        return "promote_region_flow_planner"
    if promising > 0 and float(feasibility["official_like_hard_feasible_rate"]) < 0.50:
        return "build_regional_legalizer"
    if promising > 0:
        return "refine_region_assignment_costs"
    if _max_abs_corr(corr) < 0.10:
        return "pivot_to_analytical_force_map"
    return "revisit_region_probe_assumptions"


def is_flow_safe_improvement(row: dict[str, Any]) -> bool:
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
    if (
        row["assignment_type"] == "legacy_step7g_global_report"
        or row["actual_locality_class"] == "global"
    ):
        return "route_global"
    if row["no_op"]:
        return "no_op"
    if float(row["assignment_cost"].get("fixed_preplaced_incompatibility", 0.0)) > 0.0:
        return "fixed_preplaced_conflict"
    if float(row["assignment_cost"].get("mib_group_closure_risk", 0.0)) > 0.20:
        return "MIB_group_closure_conflict"
    if float(row["assignment_cost"].get("region_slack_capacity", 0.0)) >= 1.0:
        return "no_capacity"
    if row["actual_locality_class"] not in {"regional", "macro", "local"}:
        return "no_compatible_region"
    if not bool(row["after_route_official_like_hard_feasible"]):
        return "infeasible_after_regional_edit"
    if row["official_like_cost_improving"]:
        return "none"
    return "metric_non_improving"


def _region_row(
    case: FloorSetCase, placements: dict[int, Placement], row: dict[str, Any]
) -> dict[str, Any]:
    del case, placements
    return {
        "region_id": row["region_id"],
        "row": row["row"],
        "col": row["col"],
        "xmin": row["xmin"],
        "ymin": row["ymin"],
        "xmax": row["xmax"],
        "ymax": row["ymax"],
        "area": row["area"],
        "block_area": row["block_area"],
        "block_count": row["block_count"],
        "fixed_area": row["fixed_area"],
        "preplaced_area": row["preplaced_area"],
        "boundary_area": row["boundary_area"],
        "unused_capacity": row["unused_capacity"],
        "overflow_area": row["overflow_area"],
        "utilization": row["utilization"],
    }


def _seed_groups(
    case: FloorSetCase,
    placements: dict[int, Placement],
    clusters: list[dict[str, Any]],
    *,
    group_size: int,
) -> list[dict[str, Any]]:
    movable = _movable_blocks(case)
    ranked_blocks = sorted(movable, key=lambda idx: (-_block_pressure(case, placements, idx), idx))
    groups: list[dict[str, Any]] = []
    if ranked_blocks:
        members = _expand_by_neighbors(case, placements, ranked_blocks[0], group_size, movable)
        groups.append({"assignment_type": "block_flow_assignment", "members": members})
    for cluster in clusters:
        members = [idx for idx in cluster["members"] if idx in movable]
        if len(members) >= 2:
            members = sorted(
                members, key=lambda idx: (-_block_pressure(case, placements, idx), idx)
            )[:group_size]
            groups.append({"assignment_type": "community_flow_assignment", "members": members})
            break
    bbox_members = _bbox_pressure_members(case, placements, movable, group_size)
    if bbox_members:
        groups.append({"assignment_type": "bbox_pressure_region_pull", "members": bbox_members})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for group in groups:
        key = tuple(group["members"])
        if key and key not in seen:
            unique.append(group)
            seen.add(key)
    return unique


def _select_assignments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["assignment_cost"].get("fixed_preplaced_incompatibility", 0.0)) > 0.0:
            continue
        by_type[str(row["assignment_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for _assignment_type, group in sorted(by_type.items()):
        ordered = sorted(
            group, key=lambda row: (float(row["total_cost"]), row["target_region"], row["members"])
        )
        selected.extend(ordered[:2])
    return sorted(
        selected,
        key=lambda row: (float(row["total_cost"]), row["assignment_type"], row["target_region"]),
    )


def _candidate_from_assignment(
    case_id: int,
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    assignment: dict[str, Any],
    variant_index: int,
) -> CoarseRegionAssignmentCandidate:
    t0 = time.perf_counter()
    members = list(assignment["members"])
    raw_target = [float(v) for v in assignment["target_center"]]
    target = (raw_target[0], raw_target[1])
    edited = _translate_group_to_target(baseline, members, target, frame)
    changed = tuple(sorted(_changed_blocks(baseline, edited)))
    closure = tuple(sorted(macro_closure_blocks(case, set(changed))))
    assignment_type = str(assignment["assignment_type"])
    return CoarseRegionAssignmentCandidate(
        case_id=case_id,
        candidate_id=f"case{case_id:03d}:{assignment_type}:{variant_index:03d}",
        assignment_type=assignment_type,
        descriptor_locality_class="regional",
        descriptor_repair_mode="region_repair_or_planner",
        case=case,
        baseline=baseline,
        edited=edited,
        frame=frame,
        changed_blocks=changed,
        macro_closure_blocks=closure,
        source_regions=tuple(str(rid) for rid in assignment["source_regions"]),
        target_region=str(assignment["target_region"]),
        assignment_cost={key: float(value) for key, value in assignment["assignment_cost"].items()},
        construction_status="constructed_coarse_region_assignment",
        construction_notes=(
            f"{assignment_type}: moved {len(changed)} blocks toward {assignment['target_region']}",
        ),
        construction_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _original_candidate(
    case_id: int,
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
) -> CoarseRegionAssignmentCandidate:
    return CoarseRegionAssignmentCandidate(
        case_id=case_id,
        candidate_id=f"case{case_id:03d}:original_layout",
        assignment_type="original_layout",
        descriptor_locality_class="local",
        descriptor_repair_mode="bounded_repair_pareto",
        case=case,
        baseline=baseline,
        edited=dict(baseline),
        frame=frame,
        changed_blocks=(),
        macro_closure_blocks=(),
        source_regions=(),
        target_region="original",
        assignment_cost={
            "hpwl_community_demand": 0.0,
            "region_slack_capacity": 0.0,
            "bbox_pressure": 0.0,
            "fixed_preplaced_incompatibility": 0.0,
            "mib_group_closure_risk": 0.0,
            "boundary_ownership_compatibility": 0.0,
        },
        construction_status="original_baseline",
        construction_notes=("original-inclusive baseline",),
        construction_ms=0.0,
    )


def _global_report_candidate(
    case_id: int,
    case: FloorSetCase,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
) -> CoarseRegionAssignmentCandidate:
    movable = _movable_blocks(case)
    chosen = movable[: max(1, min(len(movable), round(case.block_count * 0.80)))]
    edited = dict(baseline)
    anchor_x = frame.xmin - max(frame.width * 0.05, 1.0)
    anchor_y = frame.ymin - max(frame.height * 0.05, 1.0)
    for offset, idx in enumerate(chosen):
        _x, _y, w, h = edited[idx]
        edited[idx] = (anchor_x + (offset % 3) * 0.1, anchor_y + (offset // 3 % 3) * 0.1, w, h)
    changed = tuple(sorted(_changed_blocks(baseline, edited)))
    return CoarseRegionAssignmentCandidate(
        case_id=case_id,
        candidate_id=f"case{case_id:03d}:legacy_step7g_global_report",
        assignment_type="legacy_step7g_global_report",
        descriptor_locality_class="global",
        descriptor_repair_mode="global_route_not_local_selector",
        case=case,
        baseline=baseline,
        edited=edited,
        frame=frame,
        changed_blocks=changed,
        macro_closure_blocks=changed,
        source_regions=(),
        target_region="global_report_only",
        assignment_cost={
            "hpwl_community_demand": 1.0,
            "region_slack_capacity": 1.0,
            "bbox_pressure": 1.0,
            "fixed_preplaced_incompatibility": 0.0,
            "mib_group_closure_risk": 1.0,
            "boundary_ownership_compatibility": 1.0,
        },
        construction_status="constructed_global_report_only",
        construction_notes=("broad report-only baseline; never sent to bounded-local repair",),
        construction_ms=0.0,
    )


def _translate_group_to_target(
    placements: dict[int, Placement],
    members: list[int],
    target: tuple[float, float],
    frame: PuzzleFrame,
) -> dict[int, Placement]:
    edited = dict(placements)
    cx, cy = _group_center(placements, members)
    dx = target[0] - cx
    dy = target[1] - cy
    # Apply a bounded regional step toward the target, not a full teleport.
    max_step = max(min(frame.width, frame.height) * 0.18, 1e-9)
    norm = math.hypot(dx, dy)
    if norm > max_step:
        dx *= max_step / norm
        dy *= max_step / norm
    for idx in members:
        x, y, w, h = edited[idx]
        nx = min(max(x + dx, frame.xmin), frame.xmax - w)
        ny = min(max(y + dy, frame.ymin), frame.ymax - h)
        edited[idx] = (nx, ny, w, h)
    return edited


def _movable_blocks(case: FloorSetCase) -> list[int]:
    return [
        idx
        for idx in range(case.block_count)
        if not bool(case.constraints[idx, ConstraintColumns.FIXED].item())
        and not bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    ]


def _expand_by_neighbors(
    case: FloorSetCase,
    placements: dict[int, Placement],
    seed: int,
    group_size: int,
    movable: list[int],
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i == seed and j in movable:
            scores[j] += abs(float(weight))
        if j == seed and i in movable:
            scores[i] += abs(float(weight))
    ordered = [seed] + [
        idx
        for idx, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if idx != seed
    ]
    if len(ordered) < group_size:
        rest = sorted(
            (idx for idx in movable if idx not in set(ordered)),
            key=lambda idx: _distance(box_center(placements[seed]), box_center(placements[idx])),
        )
        ordered.extend(rest)
    return sorted(ordered[:group_size])


def _bbox_pressure_members(
    case: FloorSetCase,
    placements: dict[int, Placement],
    movable: list[int],
    group_size: int,
) -> list[int]:
    del case
    if not movable:
        return []
    xs = [(idx, box_center(placements[idx])[0]) for idx in movable]
    ys = [(idx, box_center(placements[idx])[1]) for idx in movable]
    extremes = {
        min(xs, key=lambda row: row[1])[0],
        max(xs, key=lambda row: row[1])[0],
        min(ys, key=lambda row: row[1])[0],
        max(ys, key=lambda row: row[1])[0],
    }
    center = _group_center(placements, list(placements))
    ordered = sorted(extremes, key=lambda idx: -_distance(box_center(placements[idx]), center))
    seed = ordered[0]
    return _expand_by_neighbors_no_case(placements, seed, group_size, movable)


def _expand_by_neighbors_no_case(
    placements: dict[int, Placement], seed: int, group_size: int, movable: list[int]
) -> list[int]:
    rest = sorted(
        (idx for idx in movable if idx != seed),
        key=lambda idx: _distance(box_center(placements[seed]), box_center(placements[idx])),
    )
    return sorted(([seed] + rest)[:group_size])


def _block_pressure(case: FloorSetCase, placements: dict[int, Placement], idx: int) -> float:
    x, y = box_center(placements[idx])
    total = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        other = j if i == idx else i if j == idx else None
        if other is not None and other in placements:
            ox, oy = box_center(placements[other])
            total += _distance((x, y), (ox, oy)) * abs(float(weight))
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        if int(block_idx) == idx and 0 <= int(pin_idx) < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
            total += _distance((x, y), (px, py)) * abs(float(weight))
    return total


def _group_demand_center(
    case: FloorSetCase, placements: dict[int, Placement], members: list[int]
) -> tuple[float, float]:
    points: list[tuple[float, float, float]] = []
    for idx in members:
        ex, ey = expected_block_centroid(case, idx, placements)
        points.append((ex, ey, max(_block_pressure(case, placements, idx), 1e-6)))
    total = sum(weight for _x, _y, weight in points)
    if total <= 0.0:
        return _group_center(placements, members)
    return sum(x * w for x, _y, w in points) / total, sum(y * w for _x, y, w in points) / total


def _group_center(placements: dict[int, Placement], members: list[int]) -> tuple[float, float]:
    centers = [box_center(placements[idx]) for idx in members if idx in placements]
    if not centers:
        return 0.0, 0.0
    return sum(x for x, _y in centers) / len(centers), sum(y for _x, y in centers) / len(centers)


def _bbox_pressure_cost(
    frame: PuzzleFrame, current: tuple[float, float], target: tuple[float, float]
) -> float:
    center = ((frame.xmin + frame.xmax) / 2.0, (frame.ymin + frame.ymax) / 2.0)
    before = _distance(current, center)
    after = _distance(target, center)
    diag = max(math.hypot(frame.width, frame.height), 1e-9)
    return max(after - before, 0.0) / diag


def _changed_blocks(baseline: dict[int, Placement], edited: dict[int, Placement]) -> set[int]:
    return {
        idx
        for idx, before in baseline.items()
        if idx in edited
        and any(abs(a - b) > 1e-9 for a, b in zip(before, edited[idx], strict=False))
    }


def row_with_default(region_by_id: dict[str, dict[str, Any]], target_id: str) -> dict[str, Any]:
    return region_by_id.get(target_id, {"unused_capacity": 0.0})


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "candidate_id": row["candidate_id"],
        "assignment_type": row["assignment_type"],
        "actual_locality_class": row["actual_locality_class"],
        "after_route_official_like_hard_feasible": row["after_route_official_like_hard_feasible"],
        "official_like_cost_delta": row["official_like_cost_delta"],
        "hpwl_delta": row["hpwl_delta"],
        "bbox_area_delta": row["bbox_area_delta"],
        "soft_constraint_delta": row["mib_group_boundary_soft_delta"]["total_delta"],
        "changed_block_count": row["changed_block_count"],
        "target_region": row["target_region"],
    }


def _correlation_summary(xs: list[float], ys: list[float]) -> dict[str, Any]:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"n": len(xs), "pearson": 0.0}
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        corr = 0.0
    else:
        corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / math.sqrt(vx * vy)
    return {
        "n": len(xs),
        "pearson": corr,
        "x_distribution": _distribution(xs),
        "y_distribution": _distribution(ys),
    }


def _max_abs_corr(corr: dict[str, Any]) -> float:
    values = corr.get("assignment_cost_component_correlation_with_actual_delta", {}).values()
    return max((abs(float(row.get("pearson", 0.0))) for row in values), default=0.0)


def _distribution(values: Any) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "negative_count": 0,
            "positive_count": 0,
            "zero_count": 0,
        }
    return {
        "count": len(vals),
        "min": min(vals),
        "median": median(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "negative_count": sum(int(v < -DEFAULT_EPSILON) for v in vals),
        "positive_count": sum(int(v > DEFAULT_EPSILON) for v in vals),
        "zero_count": sum(int(abs(v) <= DEFAULT_EPSILON) for v in vals),
    }


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])
