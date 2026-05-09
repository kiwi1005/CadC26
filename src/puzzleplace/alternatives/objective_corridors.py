"""Step7M objective-aligned corridor request generation.

Step7M-OAC deliberately inverts Step7L's failed flow. Heatmaps may annotate or
seed candidate centers, but every emitted request must first pass explicit
objective-vector gates for HPWL, bbox, soft roles, and full-case overlap risk.
This module is sidecar-only and does not touch contest runtime/finalizer code.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.experiments.step7l_learning_guided_replay import (
    Box,
    load_validation_cases,
    overlap_area,
    write_jsonl,
)
from puzzleplace.geometry.legality import positions_from_case_targets
from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.topology_maps import (
    GridSpec,
    build_block_heatmaps,
    terminal_weighted_centroid,
    top_cells,
)

EPS = 1e-9
DEFAULT_CASES = (19, 24, 25, 51, 76, 79, 91, 99)
GateMode = Literal["wire_safe", "bbox_shrink_wire_safe", "soft_repair_budgeted"]


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    hpwl_delta_proxy: float
    bbox_area_delta_proxy: float
    boundary_delta_proxy: float
    group_delta_proxy: float
    mib_delta_proxy: float
    overlap_risk_proxy: float
    displacement: float
    expanded_net_count: int
    heatmap_support: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlockOpportunity:
    schema: str
    case_id: str
    anchor_id: str
    block_id: int
    movable: bool
    current_box: Box
    area: float
    boundary_code: int
    has_terminal_demand: bool
    terminal_weight: float
    internal_degree: float
    current_wire_proxy: float
    current_bbox_role: str
    soft_roles: dict[str, Any]
    free_slot_count: int
    opportunity_score: float
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObjectiveCorridorRequest:
    schema: str
    case_id: str
    loader_index: int
    request_id: str
    anchor_id: str
    source_family: str
    block_id: int
    move_family: str
    target_window: dict[str, Any]
    proxy_objective_vector: dict[str, Any]
    accepted_gates: dict[str, bool]
    gate_mode: GateMode
    route_class: str
    heatmap_rank: int | None
    heatmap_score: float | None
    is_anchor: bool
    global_report_only: bool
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_opportunity_atlas(
    cases: dict[int, FloorSetCase], *, grid_size: int = 16, anchor_id: str = "original_anchor"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _case_id, case in sorted(cases.items()):
        positions = positions_from_case_targets(case)
        frame = grid_from_positions(positions, grid_size=grid_size)
        for block_id in range(case.block_count):
            rows.append(block_opportunity(case, positions, block_id, frame, anchor_id).to_json())
    return rows


def block_opportunity(
    case: FloorSetCase,
    positions: list[Box],
    block_id: int,
    frame: GridSpec,
    anchor_id: str,
) -> BlockOpportunity:
    box = positions[block_id]
    boundary_code = constraint_int(case, block_id, ConstraintColumns.BOUNDARY)
    terminal = terminal_weighted_centroid(block_id, case.p2b_edges, case.pins_pos)
    terminal_weight = float(terminal[2]) if terminal is not None else 0.0
    internal_degree = internal_weight(case, block_id)
    wire = incident_wire_proxy(case, positions, block_id, box)
    role = bbox_role(positions, block_id)
    soft = soft_role_payload(case, positions, block_id, box, frame)
    candidates = initial_candidate_boxes(case, positions, block_id, frame, grid_size=frame.rows)
    free_slots = sum(
        int(not overlaps_any(candidate, positions, skip=block_id)) for candidate in candidates
    )
    movable = is_movable(case, block_id)
    score = (wire + terminal_weight + internal_degree) * (1.0 if movable else 0.0)
    if role != "interior":
        score *= 1.15
    return BlockOpportunity(
        schema="step7m_phase0_block_opportunity_v1",
        case_id=str(case.case_id),
        anchor_id=anchor_id,
        block_id=block_id,
        movable=movable,
        current_box=box,
        area=box[2] * box[3],
        boundary_code=boundary_code,
        has_terminal_demand=terminal is not None,
        terminal_weight=terminal_weight,
        internal_degree=internal_degree,
        current_wire_proxy=wire,
        current_bbox_role=role,
        soft_roles=soft,
        free_slot_count=free_slots,
        opportunity_score=score,
        provenance={
            "source": "validation_replay_anchor_geometry",
            "label_policy": "anchor geometry used only for sidecar objective-corridor diagnostics",
            "grid": {"rows": frame.rows, "cols": frame.cols},
        },
    )


def generate_corridor_requests(
    cases: dict[int, FloorSetCase],
    *,
    grid_size: int = 16,
    candidate_cells_per_block: int = 24,
    max_blocks_per_case: int = 8,
    windows_per_block: int = 4,
    gate_modes: Iterable[GateMode] = ("wire_safe", "bbox_shrink_wire_safe", "soft_repair_budgeted"),
    anchor_id: str = "original_anchor",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[ObjectiveCorridorRequest] = []
    rejection_counts: Counter[str] = Counter()
    for loader_index, case in sorted(cases.items()):
        positions = positions_from_case_targets(case)
        frame = grid_from_positions(positions, grid_size=grid_size)
        opportunities = [
            block_opportunity(case, positions, block_id, frame, anchor_id)
            for block_id in range(case.block_count)
        ]
        ranked_blocks = select_corridor_opportunities(
            opportunities, max_blocks_per_case=max_blocks_per_case
        )
        accepted_by_block: dict[int, list[ObjectiveCorridorRequest]] = defaultdict(list)
        for opportunity in ranked_blocks:
            candidates = objective_corridor_candidates_for_block(
                case,
                positions,
                opportunity.block_id,
                frame,
                candidate_cells_per_block=candidate_cells_per_block,
            )
            for candidate in candidates:
                vector = objective_vector_for_candidate(
                    case, positions, opportunity.block_id, candidate["box"], frame, candidate
                )
                accepted_any = False
                for gate_mode in gate_modes:
                    gates = accepted_gates(vector, mode=gate_mode)
                    if all(gates.values()):
                        accepted_any = True
                        request = corridor_request(
                            case=case,
                            loader_index=loader_index,
                            block_id=opportunity.block_id,
                            target_box=candidate["box"],
                            frame=frame,
                            vector=vector,
                            gates=gates,
                            gate_mode=gate_mode,
                            source_family=str(candidate["source_family"]),
                            heatmap_rank=candidate.get("heatmap_rank"),
                            anchor_id=anchor_id,
                        )
                        accepted_by_block[opportunity.block_id].append(request)
                if not accepted_any:
                    rejection_counts.update(rejection_reasons(vector))
        rows.extend(_cap_and_dedupe_requests(accepted_by_block, windows_per_block))
    json_rows = [row.to_json() for row in rows]
    summary = summarize_corridor_requests(json_rows, rejection_counts=rejection_counts)
    return json_rows, summary


def select_corridor_opportunities(
    opportunities: list[BlockOpportunity], *, max_blocks_per_case: int
) -> list[BlockOpportunity]:
    """Select a small, diverse block deck for objective-corridor probing."""

    movable = [row for row in opportunities if row.movable and row.free_slot_count > 0]
    if not movable or max_blocks_per_case <= 0:
        return []
    high_attribution = sorted(
        movable, key=lambda row: (-row.opportunity_score, -row.free_slot_count, row.block_id)
    )[: max(2, max_blocks_per_case // 4)]
    low_risk_plateau = sorted(
        movable,
        key=lambda row: (
            row.current_wire_proxy + row.terminal_weight + row.internal_degree,
            -row.free_slot_count,
            row.block_id,
        ),
    )
    hull_shrink = sorted(
        [row for row in movable if row.current_bbox_role != "interior"],
        key=lambda row: (-row.opportunity_score, row.block_id),
    )
    soft_repair = sorted(
        [
            row
            for row in movable
            if row.soft_roles.get("boundary_code")
            or row.soft_roles.get("cluster_id")
            or row.soft_roles.get("mib_id")
        ],
        key=lambda row: (-row.opportunity_score, row.block_id),
    )

    selected: list[BlockOpportunity] = []
    seen: set[int] = set()
    for bucket in (high_attribution, low_risk_plateau, hull_shrink, soft_repair):
        for row in bucket:
            if row.block_id in seen:
                continue
            selected.append(row)
            seen.add(row.block_id)
            if len(selected) >= max_blocks_per_case:
                return selected
    return selected


def objective_corridor_candidates_for_block(
    case: FloorSetCase,
    positions: list[Box],
    block_id: int,
    frame: GridSpec,
    *,
    candidate_cells_per_block: int,
) -> list[dict[str, Any]]:
    current = positions[block_id]
    centers: list[dict[str, Any]] = []
    centers.extend(axis_micro_corridor_centers(current, frame))
    centers.extend(local_neighborhood_centers(current, frame))
    centers.extend(terminal_centers(case, block_id, frame))
    centers.extend(bbox_inward_centers(positions, block_id, frame))
    centers.extend(heatmap_centers(case, block_id, frame, k=max(4, candidate_cells_per_block // 4)))
    deduped = dedupe_centers(centers)[:candidate_cells_per_block]
    out: list[dict[str, Any]] = []
    for item in deduped:
        box = nearest_nonoverlap_box(
            positions,
            block_id,
            target_center=(float(item["cx"]), float(item["cy"])),
            frame=frame,
        )
        if box is None or same_box(box, current):
            continue
        out.append({**item, "box": box})
    return out


def corridor_request(
    *,
    case: FloorSetCase,
    loader_index: int,
    block_id: int,
    target_box: Box,
    frame: GridSpec,
    vector: ObjectiveVector,
    gates: dict[str, bool],
    gate_mode: GateMode,
    source_family: str,
    heatmap_rank: int | None,
    anchor_id: str,
) -> ObjectiveCorridorRequest:
    case_id = str(case.case_id)
    x, y, w, h = target_box
    row, col = frame.cell_for_point(x + w / 2.0, y + h / 2.0)
    request_id = (
        f"step7m_oac_case{case_id}_{gate_mode}_{source_family}"
        f"_b{block_id}_r{row}c{col}_x{round(x, 3)}_y{round(y, 3)}"
    )
    route_class = "unrouted_objective_corridor_sidecar"
    if gate_mode == "soft_repair_budgeted":
        route_class = "soft_budgeted_objective_corridor_sidecar"
    return ObjectiveCorridorRequest(
        schema="step7m_phase1_objective_corridor_request_v1",
        case_id=case_id,
        loader_index=loader_index,
        request_id=request_id,
        anchor_id=anchor_id,
        source_family=source_family,
        block_id=block_id,
        move_family=move_family(case, block_id, gate_mode),
        target_window=window_from_box(target_box, frame),
        proxy_objective_vector=vector.to_json(),
        accepted_gates=gates,
        gate_mode=gate_mode,
        route_class=route_class,
        heatmap_rank=heatmap_rank,
        heatmap_score=vector.heatmap_support,
        is_anchor=False,
        global_report_only=False,
        provenance={
            "source": "step7m_objective_aligned_corridor",
            "anchor_geometry_source": "validation_replay_anchor_geometry",
            "label_policy": "sidecar proxy generation only; no model training labels consumed",
            "heatmap_used_as": "candidate_source_and_tiebreak_only",
        },
    )


def summarize_opportunity_atlas(rows: list[dict[str, Any]], *, grid_size: int) -> dict[str, Any]:
    cases = {str(row["case_id"]) for row in rows}
    movable = [row for row in rows if row.get("movable")]
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row["case_id"])]["total"] += 1
        if row.get("movable"):
            by_case[str(row["case_id"])]["movable"] += 1
        if int(row.get("boundary_code", 0)) != 0:
            by_case[str(row["case_id"])]["boundary"] += 1
        if row.get("has_terminal_demand"):
            by_case[str(row["case_id"])]["terminal"] += 1
    return {
        "schema": "step7m_phase0_opportunity_summary_v1",
        "decision": "promote_to_objective_corridor_requests" if movable else "fix_empty_atlas",
        "grid": grid_size,
        "case_count": len(cases),
        "row_count": len(rows),
        "movable_row_count": len(movable),
        "case025_row_share": _case_share(rows, "25"),
        "case025_movable_share": _case_share(movable, "25"),
        "free_slot_positive_count": sum(int(row.get("free_slot_count", 0) > 0) for row in rows),
        "per_case_counts": {case_id: dict(counter) for case_id, counter in sorted(by_case.items())},
        "forbidden_validation_label_terms": forbidden_term_count(rows),
    }


def summarize_corridor_requests(
    rows: list[dict[str, Any]], *, rejection_counts: Counter[str] | None = None
) -> dict[str, Any]:
    rejection_counts = rejection_counts or Counter()
    per_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_case[str(row["case_id"])][str(row["gate_mode"])] += 1
    unique = {request_signature(row) for row in rows}
    wire_rows = [row for row in rows if row.get("gate_mode") == "wire_safe"]
    represented = {str(row["case_id"]) for row in rows}
    gate_counts = Counter(str(row["gate_mode"]) for row in rows)
    hpwl_regressions = sum(
        int(row["proxy_objective_vector"]["hpwl_delta_proxy"] > EPS) for row in wire_rows
    )
    bbox_regressions = sum(
        int(row["proxy_objective_vector"]["bbox_area_delta_proxy"] > EPS) for row in wire_rows
    )
    soft_regressions = sum(int(_soft_sum(row["proxy_objective_vector"]) > EPS) for row in wire_rows)
    forbidden_terms = forbidden_term_count(rows)
    case025_share = _case_share(rows, "25")
    soft_budget_share = gate_counts.get("soft_repair_budgeted", 0) / max(len(rows), 1)
    decision = "promote_to_objective_guarded_replay"
    if (
        len(unique) < 64
        or len(represented) < 6
        or case025_share > 0.35
        or soft_budget_share > 0.50
        or hpwl_regressions > 0
        or bbox_regressions > 0
        or soft_regressions > 0
        or forbidden_terms > 0
    ):
        decision = "fix_objective_corridor_coverage"
    return {
        "schema": "step7m_phase1_corridor_request_summary_v1",
        "decision": decision,
        "request_count": len(rows),
        "unique_request_signature_count": len(unique),
        "represented_case_count": len(represented),
        "case025_request_share": case025_share,
        "soft_budgeted_request_share": soft_budget_share,
        "request_count_by_gate_mode": dict(gate_counts),
        "request_count_by_source_family": dict(Counter(str(row["source_family"]) for row in rows)),
        "per_case_request_count_by_gate_mode": {
            case_id: dict(counter) for case_id, counter in sorted(per_case.items())
        },
        "rejection_counts": dict(rejection_counts),
        "predicted_hpwl_regression_count_wire_safe": hpwl_regressions,
        "predicted_bbox_regression_count_wire_safe": bbox_regressions,
        "predicted_soft_regression_count_wire_safe": soft_regressions,
        "heatmap_supported_accepted_count": sum(
            int(row.get("heatmap_rank") is not None) for row in rows
        ),
        "forbidden_validation_label_terms": forbidden_terms,
    }


def write_opportunity_artifacts(
    rows: list[dict[str, Any]], out_path: Path, summary_path: Path, *, grid_size: int
) -> dict[str, Any]:
    count = write_jsonl(out_path, rows)
    summary = summarize_opportunity_atlas(rows, grid_size=grid_size)
    summary["row_count"] = count
    summary["out_path"] = str(out_path)
    write_json(summary_path, summary)
    summary_path.with_suffix(".md").write_text(
        opportunity_summary_markdown(summary), encoding="utf-8"
    )
    return summary


def write_corridor_artifacts(
    rows: list[dict[str, Any]], summary: dict[str, Any], out_path: Path, summary_path: Path
) -> dict[str, Any]:
    count = write_jsonl(out_path, rows)
    summary = {**summary, "request_count": count, "out_path": str(out_path)}
    write_json(summary_path, summary)
    summary_path.with_suffix(".md").write_text(corridor_summary_markdown(summary), encoding="utf-8")
    return summary


def load_cases_for_step7m(
    base_dir: Path, validation_cases: Iterable[int]
) -> dict[int, FloorSetCase]:
    return load_validation_cases(base_dir, sorted({int(case) for case in validation_cases}))


def grid_from_positions(
    positions: list[Box], *, grid_size: int = 16, pad_fraction: float = 0.08
) -> GridSpec:
    x0 = min(x for x, _y, _w, _h in positions)
    y0 = min(y for _x, y, _w, _h in positions)
    x1 = max(x + w for x, _y, w, _h in positions)
    y1 = max(y + h for _x, y, _w, h in positions)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    pad = max(width, height) * pad_fraction
    return GridSpec(
        rows=grid_size,
        cols=grid_size,
        x=x0 - pad,
        y=y0 - pad,
        w=width + 2.0 * pad,
        h=height + 2.0 * pad,
    )


def incident_wire_proxy(
    case: FloorSetCase, positions: list[Box], block_id: int, new_box: Box
) -> float:
    centers = [_box_center(box) for box in positions]
    centers[block_id] = _box_center(new_box)
    total = 0.0
    for edge in case.b2b_edges:
        if edge.numel() < 2 or float(edge[0].item()) < 0:
            continue
        src = int(edge[0].item())
        dst = int(edge[1].item())
        if block_id not in (src, dst) or src >= len(centers) or dst >= len(centers):
            continue
        weight = _edge_weight(edge)
        total += weight * manhattan(centers[src], centers[dst])
    for edge in case.p2b_edges:
        if edge.numel() < 2 or float(edge[0].item()) < 0:
            continue
        pin = int(edge[0].item())
        dst = int(edge[1].item())
        if dst != block_id or pin < 0 or pin >= int(case.pins_pos.shape[0]):
            continue
        weight = _edge_weight(edge)
        total += weight * manhattan(centers[block_id], _pin_center(case, pin))
    return total


def internal_weight(case: FloorSetCase, block_id: int) -> float:
    total = 0.0
    for edge in case.b2b_edges:
        if edge.numel() < 2 or float(edge[0].item()) < 0:
            continue
        if block_id in (int(edge[0].item()), int(edge[1].item())):
            total += _edge_weight(edge)
    return total


def objective_vector_for_candidate(
    case: FloorSetCase,
    positions: list[Box],
    block_id: int,
    target_box: Box,
    frame: GridSpec,
    candidate: dict[str, Any],
) -> ObjectiveVector:
    current = positions[block_id]
    current_wire = incident_wire_proxy(case, positions, block_id, current)
    new_wire = incident_wire_proxy(case, positions, block_id, target_box)
    bbox_delta = layout_bbox_area(replace_box(positions, block_id, target_box)) - layout_bbox_area(
        positions
    )
    soft = soft_delta_proxy(case, positions, block_id, target_box, frame)
    overlap_risk = float(overlaps_any(target_box, positions, skip=block_id))
    return ObjectiveVector(
        hpwl_delta_proxy=new_wire - current_wire,
        bbox_area_delta_proxy=bbox_delta,
        boundary_delta_proxy=soft["boundary_delta_proxy"],
        group_delta_proxy=soft["group_delta_proxy"],
        mib_delta_proxy=soft["mib_delta_proxy"],
        overlap_risk_proxy=overlap_risk,
        displacement=manhattan(_box_center(current), _box_center(target_box)),
        expanded_net_count=count_expanded_incident_edges(case, positions, block_id, target_box),
        heatmap_support=candidate.get("heatmap_score"),
    )


def accepted_gates(vector: ObjectiveVector, *, mode: GateMode) -> dict[str, bool]:
    soft_ok = (
        vector.boundary_delta_proxy <= EPS
        and vector.group_delta_proxy <= EPS
        and vector.mib_delta_proxy <= EPS
    )
    base = {
        "overlap": vector.overlap_risk_proxy <= EPS,
        "hpwl": vector.hpwl_delta_proxy <= EPS,
        "bbox": vector.bbox_area_delta_proxy <= EPS,
        "soft": soft_ok,
    }
    if mode == "bbox_shrink_wire_safe":
        base["bbox"] = vector.bbox_area_delta_proxy < -EPS
    if mode == "soft_repair_budgeted":
        soft_sum = vector.boundary_delta_proxy + vector.group_delta_proxy + vector.mib_delta_proxy
        base["soft"] = soft_sum < -EPS
        base["hpwl"] = vector.hpwl_delta_proxy <= max(1.0, vector.displacement * 0.05)
        base["bbox"] = vector.bbox_area_delta_proxy <= EPS
    return base


def rejection_reasons(vector: ObjectiveVector) -> list[str]:
    reasons = []
    if vector.overlap_risk_proxy > EPS:
        reasons.append("overlap_risk")
    if vector.hpwl_delta_proxy > EPS:
        reasons.append("hpwl_regression_proxy")
    if vector.bbox_area_delta_proxy > EPS:
        reasons.append("bbox_regression_proxy")
    if vector.boundary_delta_proxy > EPS:
        reasons.append("boundary_regression_proxy")
    if vector.group_delta_proxy > EPS:
        reasons.append("group_regression_proxy")
    if vector.mib_delta_proxy > EPS:
        reasons.append("mib_regression_proxy")
    return reasons or ["dominated_or_duplicate"]


def initial_candidate_boxes(
    case: FloorSetCase, positions: list[Box], block_id: int, frame: GridSpec, *, grid_size: int
) -> list[Box]:
    del case, grid_size
    current = positions[block_id]
    boxes = []
    for center in local_neighborhood_centers(current, frame) + bbox_inward_centers(
        positions, block_id, frame
    ):
        candidate = nearest_nonoverlap_box(
            positions,
            block_id,
            target_center=(float(center["cx"]), float(center["cy"])),
            frame=frame,
        )
        if candidate is not None:
            boxes.append(candidate)
    return boxes


def nearest_nonoverlap_box(
    positions: list[Box], block_id: int, *, target_center: tuple[float, float], frame: GridSpec
) -> Box | None:
    current = positions[block_id]
    _x, _y, w, h = current
    step_x = max(frame.w / max(frame.cols, 1), w / 2.0, 1.0)
    step_y = max(frame.h / max(frame.rows, 1), h / 2.0, 1.0)
    base_x = target_center[0] - w / 2.0
    base_y = target_center[1] - h / 2.0
    candidates: list[Box] = []
    for radius in range(0, 4):
        for dy_i in range(-radius, radius + 1):
            for dx_i in range(-radius, radius + 1):
                if radius and max(abs(dx_i), abs(dy_i)) != radius:
                    continue
                candidates.append((base_x + dx_i * step_x, base_y + dy_i * step_y, w, h))
    candidates.extend(frame_corner_boxes(frame, w, h))
    feasible = [box for box in candidates if not overlaps_any(box, positions, skip=block_id)]
    feasible = [box for box in feasible if not same_box(box, current)]
    feasible.sort(
        key=lambda box: (
            manhattan(_box_center(box), target_center),
            manhattan(_box_center(box), _box_center(current)),
        )
    )
    return feasible[0] if feasible else None


def axis_micro_corridor_centers(box: Box, frame: GridSpec) -> list[dict[str, Any]]:
    """Tiny axis-aligned probes for local HPWL/bbox plateaus.

    These centers are not allowed to bypass objective gates; they only make the
    candidate sampler sensitive to piecewise-linear non-regression directions
    that a one-cell jump can skip over.
    """

    cx, cy = _box_center(box)
    grid_step = min(frame.w / max(frame.cols, 1), frame.h / max(frame.rows, 1))
    step = max(min(grid_step * 0.05, max(box[2], box[3]) * 0.25), 1e-3)
    centers: list[dict[str, Any]] = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        delta = step * scale
        centers.extend(
            [
                {"cx": cx - delta, "cy": cy, "source_family": "micro_axis_corridor"},
                {"cx": cx + delta, "cy": cy, "source_family": "micro_axis_corridor"},
                {"cx": cx, "cy": cy - delta, "source_family": "micro_axis_corridor"},
                {"cx": cx, "cy": cy + delta, "source_family": "micro_axis_corridor"},
            ]
        )
    return centers


def local_neighborhood_centers(box: Box, frame: GridSpec) -> list[dict[str, Any]]:
    cx, cy = _box_center(box)
    step_x = frame.w / max(frame.cols, 1)
    step_y = frame.h / max(frame.rows, 1)
    centers = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        centers.append(
            {"cx": cx + dx * step_x, "cy": cy + dy * step_y, "source_family": "local_corridor"}
        )
    return centers


def terminal_centers(case: FloorSetCase, block_id: int, frame: GridSpec) -> list[dict[str, Any]]:
    centroid = terminal_weighted_centroid(block_id, case.p2b_edges, case.pins_pos)
    if centroid is None:
        return []
    return [{"cx": centroid[0], "cy": centroid[1], "source_family": "terminal_corridor"}]


def bbox_inward_centers(
    positions: list[Box], block_id: int, frame: GridSpec
) -> list[dict[str, Any]]:
    box = positions[block_id]
    cx, cy = _box_center(box)
    step_x = frame.w / max(frame.cols, 1)
    step_y = frame.h / max(frame.rows, 1)
    role = bbox_role(positions, block_id)
    out = []
    if "left" in role:
        out.append({"cx": cx + step_x, "cy": cy, "source_family": "bbox_inward_corridor"})
    if "right" in role:
        out.append({"cx": cx - step_x, "cy": cy, "source_family": "bbox_inward_corridor"})
    if "bottom" in role:
        out.append({"cx": cx, "cy": cy + step_y, "source_family": "bbox_inward_corridor"})
    if "top" in role:
        out.append({"cx": cx, "cy": cy - step_y, "source_family": "bbox_inward_corridor"})
    return out


def heatmap_centers(
    case: FloorSetCase, block_id: int, frame: GridSpec, *, k: int
) -> list[dict[str, Any]]:
    maps = build_block_heatmaps(
        block_id=block_id,
        boundary_code=constraint_int(case, block_id, ConstraintColumns.BOUNDARY),
        p2b_edges=case.p2b_edges,
        pins_pos=case.pins_pos,
        grid=frame,
    )
    out = []
    for rank, cell in enumerate(top_cells(maps["topology"], k=k)):
        cx, cy = frame.cell_center(int(cell["row"]), int(cell["col"]))
        out.append(
            {
                "cx": cx,
                "cy": cy,
                "source_family": "heatmap_tiebreak_corridor",
                "heatmap_rank": rank,
                "heatmap_score": float(cell["score"]),
            }
        )
    return out


def soft_delta_proxy(
    case: FloorSetCase, positions: list[Box], block_id: int, target_box: Box, frame: GridSpec
) -> dict[str, float]:
    current = positions[block_id]
    return {
        "boundary_delta_proxy": boundary_distance(case, block_id, target_box, frame)
        - boundary_distance(case, block_id, current, frame),
        "group_delta_proxy": peer_distance(
            case, positions, block_id, target_box, ConstraintColumns.CLUSTER
        )
        - peer_distance(case, positions, block_id, current, ConstraintColumns.CLUSTER),
        "mib_delta_proxy": peer_distance(
            case, positions, block_id, target_box, ConstraintColumns.MIB
        )
        - peer_distance(case, positions, block_id, current, ConstraintColumns.MIB),
    }


def soft_role_payload(
    case: FloorSetCase, positions: list[Box], block_id: int, box: Box, frame: GridSpec
) -> dict[str, Any]:
    return {
        "boundary_code": constraint_int(case, block_id, ConstraintColumns.BOUNDARY),
        "cluster_id": constraint_int(case, block_id, ConstraintColumns.CLUSTER),
        "mib_id": constraint_int(case, block_id, ConstraintColumns.MIB),
        "boundary_distance": boundary_distance(case, block_id, box, frame),
        "cluster_peer_distance": peer_distance(
            case, positions, block_id, box, ConstraintColumns.CLUSTER
        ),
        "mib_peer_distance": peer_distance(case, positions, block_id, box, ConstraintColumns.MIB),
    }


def boundary_distance(case: FloorSetCase, block_id: int, box: Box, frame: GridSpec) -> float:
    code = constraint_int(case, block_id, ConstraintColumns.BOUNDARY)
    if code <= 0:
        return 0.0
    x, y, w, h = box
    distances = []
    if code & 1:
        distances.append(abs(x - frame.x))
    if code & 2:
        distances.append(abs((x + w) - (frame.x + frame.w)))
    if code & 4:
        distances.append(abs((y + h) - (frame.y + frame.h)))
    if code & 8:
        distances.append(abs(y - frame.y))
    return min(distances) if distances else 0.0


def peer_distance(
    case: FloorSetCase, positions: list[Box], block_id: int, box: Box, column: ConstraintColumns
) -> float:
    group_id = constraint_int(case, block_id, column)
    if group_id <= 0:
        return 0.0
    peers = [
        idx
        for idx in range(case.block_count)
        if idx != block_id and constraint_int(case, idx, column) == group_id
    ]
    if not peers:
        return 0.0
    center = _box_center(box)
    return sum(manhattan(center, _box_center(positions[idx])) for idx in peers) / len(peers)


def overlaps_any(candidate: Box, positions: list[Box], *, skip: int) -> bool:
    return any(
        index != skip and overlap_area(candidate, other) > EPS
        for index, other in enumerate(positions)
    )


def count_expanded_incident_edges(
    case: FloorSetCase, positions: list[Box], block_id: int, target_box: Box
) -> int:
    current = incident_wire_proxy(case, positions, block_id, positions[block_id])
    new = incident_wire_proxy(case, positions, block_id, target_box)
    return int(new > current + EPS)


def bbox_role(positions: list[Box], block_id: int) -> str:
    box = positions[block_id]
    x0 = min(x for x, _y, _w, _h in positions)
    y0 = min(y for _x, y, _w, _h in positions)
    x1 = max(x + w for x, _y, w, _h in positions)
    y1 = max(y + h for _x, y, _w, h in positions)
    roles = []
    if abs(box[0] - x0) <= EPS:
        roles.append("left")
    if abs(box[0] + box[2] - x1) <= EPS:
        roles.append("right")
    if abs(box[1] - y0) <= EPS:
        roles.append("bottom")
    if abs(box[1] + box[3] - y1) <= EPS:
        roles.append("top")
    return "+".join(roles) if roles else "interior"


def layout_bbox_area(positions: list[Box]) -> float:
    x0 = min(x for x, _y, _w, _h in positions)
    y0 = min(y for _x, y, _w, _h in positions)
    x1 = max(x + w for x, _y, w, _h in positions)
    y1 = max(y + h for _x, y, _w, h in positions)
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def replace_box(positions: list[Box], block_id: int, box: Box) -> list[Box]:
    out = list(positions)
    out[block_id] = box
    return out


def frame_corner_boxes(frame: GridSpec, w: float, h: float) -> list[Box]:
    return [
        (frame.x, frame.y, w, h),
        (frame.x + max(frame.w - w, 0.0), frame.y, w, h),
        (frame.x, frame.y + max(frame.h - h, 0.0), w, h),
        (frame.x + max(frame.w - w, 0.0), frame.y + max(frame.h - h, 0.0), w, h),
    ]


def window_from_box(box: Box, frame: GridSpec) -> dict[str, Any]:
    x, y, w, h = box
    row, col = frame.cell_for_point(x + w / 2.0, y + h / 2.0)
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": x + w / 2.0,
        "cy": y + h / 2.0,
        "row": row,
        "col": col,
        "grid": {"rows": frame.rows, "cols": frame.cols},
        "frame": {"x": frame.x, "y": frame.y, "w": frame.w, "h": frame.h},
    }


def move_family(case: FloorSetCase, block_id: int, gate_mode: GateMode) -> str:
    if constraint_int(case, block_id, ConstraintColumns.BOUNDARY) > 0:
        return f"boundary_{gate_mode}"
    if constraint_int(case, block_id, ConstraintColumns.CLUSTER) > 0:
        return f"group_{gate_mode}"
    if constraint_int(case, block_id, ConstraintColumns.MIB) > 0:
        return f"mib_{gate_mode}"
    return f"single_block_{gate_mode}"


def is_movable(case: FloorSetCase, block_id: int) -> bool:
    return not bool(case.constraints[block_id, ConstraintColumns.FIXED].item()) and not bool(
        case.constraints[block_id, ConstraintColumns.PREPLACED].item()
    )


def constraint_int(case: FloorSetCase, block_id: int, column: ConstraintColumns) -> int:
    if block_id < 0 or block_id >= case.block_count or case.constraints.shape[-1] <= int(column):
        return 0
    return int(float(case.constraints[block_id, column].item()))


def manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _box_center(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _pin_center(case: FloorSetCase, pin_id: int) -> tuple[float, float]:
    return float(case.pins_pos[pin_id, 0].item()), float(case.pins_pos[pin_id, 1].item())


def _edge_weight(edge: Any) -> float:
    return max(float(edge[2].item()) if edge.numel() > 2 else 1.0, 0.0)


def same_box(a: Box, b: Box) -> bool:
    return all(abs(x - y) <= EPS for x, y in zip(a, b, strict=False))


def dedupe_centers(centers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[float, float]] = set()
    out = []
    for center in centers:
        key = (round(float(center["cx"]), 3), round(float(center["cy"]), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(center)
    return out


def _cap_and_dedupe_requests(
    requests_by_block: dict[int, list[ObjectiveCorridorRequest]], windows_per_block: int
) -> list[ObjectiveCorridorRequest]:
    out: list[ObjectiveCorridorRequest] = []
    seen: set[str] = set()
    for _block_id, requests in sorted(requests_by_block.items()):
        requests.sort(
            key=lambda row: (
                row.proxy_objective_vector["hpwl_delta_proxy"],
                row.proxy_objective_vector["bbox_area_delta_proxy"],
                _soft_sum(row.proxy_objective_vector),
                row.proxy_objective_vector["displacement"],
                row.heatmap_rank if row.heatmap_rank is not None else 10**6,
            )
        )
        kept = 0
        for request in requests:
            signature = request_signature(request.to_json())
            if signature in seen:
                continue
            seen.add(signature)
            out.append(request)
            kept += 1
            if kept >= windows_per_block:
                break
    return out


def request_signature(row: dict[str, Any]) -> str:
    window = row.get("target_window") or {}
    return (
        f"case={row.get('case_id')}|gate={row.get('gate_mode')}|block={row.get('block_id')}|"
        f"x={round(float(window.get('x', 0.0)), 3)}|y={round(float(window.get('y', 0.0)), 3)}"
    )


def forbidden_term_count(rows: list[dict[str, Any]]) -> int:
    forbidden = ("fp_sol", "polygons", "metrics", "target_positions", "oracle_layout")
    count = 0
    for row in rows:
        text = json.dumps(row, sort_keys=True).lower()
        count += int(any(term in text for term in forbidden))
    return count


def opportunity_summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7M Phase 0 Opportunity Atlas

Decision: `{summary["decision"]}`

- case_count: {summary["case_count"]}
- row_count: {summary["row_count"]}
- movable_row_count: {summary["movable_row_count"]}
- free_slot_positive_count: {summary["free_slot_positive_count"]}
- case025_row_share: {summary["case025_row_share"]:.3f}
- case025_movable_share: {summary["case025_movable_share"]:.3f}
- forbidden_validation_label_terms: {summary["forbidden_validation_label_terms"]}
"""


def corridor_summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# Step7M Phase 1 Objective Corridor Requests

Decision: `{summary["decision"]}`

- request_count: {summary["request_count"]}
- unique_request_signature_count: {summary["unique_request_signature_count"]}
- represented_case_count: {summary["represented_case_count"]}
- case025_request_share: {summary["case025_request_share"]:.3f}
- request_count_by_gate_mode: {summary["request_count_by_gate_mode"]}
- request_count_by_source_family: {summary["request_count_by_source_family"]}
- rejection_counts: {summary["rejection_counts"]}
- predicted_hpwl_regression_count_wire_safe: {summary["predicted_hpwl_regression_count_wire_safe"]}
- predicted_bbox_regression_count_wire_safe: {summary["predicted_bbox_regression_count_wire_safe"]}
- predicted_soft_regression_count_wire_safe: {summary["predicted_soft_regression_count_wire_safe"]}
- forbidden_validation_label_terms: {summary["forbidden_validation_label_terms"]}
"""


def _case_share(rows: list[dict[str, Any]], case_id: str) -> float:
    if not rows:
        return 0.0
    return sum(int(str(row.get("case_id")) == case_id) for row in rows) / len(rows)


def _soft_sum(vector: dict[str, Any]) -> float:
    return (
        float(vector.get("boundary_delta_proxy", 0.0))
        + float(vector.get("group_delta_proxy", 0.0))
        + float(vector.get("mib_delta_proxy", 0.0))
    )
