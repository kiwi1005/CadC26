from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.region_topology import (
    block_area,
    build_region_grid,
    cell_overlap_area,
    point_region,
)
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame


def build_locality_maps(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    resolutions: tuple[tuple[str, int, int], ...] | None = None,
) -> dict[str, Any]:
    if resolutions is None:
        adaptive = max(4, min(12, int(math.ceil(math.sqrt(case.block_count)))))
        resolutions = (("coarse", 4, 4), ("adaptive", adaptive, adaptive))
    maps = [
        build_resolution_locality_map(case, placements, frame, name=name, rows=rows, cols=cols)
        for name, rows, cols in resolutions
    ]
    return {
        "case_id": case.case_id,
        "block_count": case.block_count,
        "resolutions": maps,
        "sensitivity": locality_sensitivity(maps),
    }


def build_resolution_locality_map(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    name: str,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    cells = build_region_grid(frame, rows=rows, cols=cols)
    total_block_area = sum(block_area(box) for box in placements.values())
    pin_counts = _pin_counts(case, frame, rows=rows, cols=cols)
    net_demand = _net_demand(case, placements, frame, rows=rows, cols=cols)
    out: list[dict[str, Any]] = []
    for cell in cells:
        occupancy = 0.0
        fixed_preplaced = 0.0
        boundary_owner = 0.0
        mib_group = 0.0
        for idx, box in placements.items():
            overlap = cell_overlap_area(box, cell)
            if overlap <= 0.0:
                continue
            occupancy += overlap
            if bool(case.constraints[idx, ConstraintColumns.FIXED].item()) or bool(
                case.constraints[idx, ConstraintColumns.PREPLACED].item()
            ):
                fixed_preplaced += overlap
            if bool(case.constraints[idx, ConstraintColumns.BOUNDARY].item()):
                boundary_owner += overlap
            if bool(case.constraints[idx, ConstraintColumns.MIB].item()) or bool(
                case.constraints[idx, ConstraintColumns.CLUSTER].item()
            ):
                mib_group += overlap
        area = max(float(cell["area"]), 1e-9)
        occupancy_ratio = occupancy / area
        slack = max(area - occupancy, 0.0)
        hole_score = max(1.0 - occupancy_ratio, 0.0)
        repair_reachability = _repair_reachability(occupancy_ratio, mib_group / area, slack, area)
        rid = str(cell["region_id"])
        out.append(
            {
                **cell,
                "occupancy_mask": occupancy_ratio,
                "free_space_mask": slack / area,
                "fixed_preplaced_mask": fixed_preplaced / area,
                "pin_density_heatmap": pin_counts.get(rid, 0) / area,
                "net_community_demand_map": net_demand.get(rid, 0.0) / max(total_block_area, 1e-9),
                "region_slack_map": slack,
                "hole_fragmentation_map": hole_score,
                "boundary_owner_map": boundary_owner / area,
                "MIB_group_closure_mask": mib_group / area,
                "repair_reachability_mask": repair_reachability,
            }
        )
    return {
        "name": name,
        "grid": {"rows": rows, "cols": cols},
        "regions": out,
        "summary": _map_summary(out),
    }


def locality_sensitivity(maps: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in maps}
    if "coarse" not in by_name or "adaptive" not in by_name:
        return {}
    coarse = by_name["coarse"]["summary"]
    adaptive = by_name["adaptive"]["summary"]
    keys = (
        "max_occupancy",
        "mean_free_space",
        "max_pin_density",
        "max_repair_reachability",
    )
    return {
        f"{key}_delta_adaptive_minus_coarse": float(adaptive.get(key, 0.0))
        - float(coarse.get(key, 0.0))
        for key in keys
    }


def touched_region_stats(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    touched_blocks: set[int],
    *,
    rows: int = 4,
    cols: int = 4,
) -> dict[str, Any]:
    regions = {
        point_region(*_center(placements[idx]), frame, rows=rows, cols=cols)
        for idx in touched_blocks
        if idx in placements
    }
    macro_closure = macro_closure_blocks(case, touched_blocks)
    return {
        "touched_region_count": len(regions),
        "touched_regions": sorted(regions),
        "macro_closure_size": len(macro_closure),
        "macro_closure_fraction": len(macro_closure) / max(case.block_count, 1),
    }


def macro_closure_blocks(case: FloorSetCase, seed: set[int]) -> set[int]:
    out = set(seed)
    for column in (ConstraintColumns.MIB, ConstraintColumns.CLUSTER):
        groups: dict[int, set[int]] = {}
        for idx in range(case.block_count):
            value = int(float(case.constraints[idx, column].item()))
            if value > 0:
                groups.setdefault(value, set()).add(idx)
        for members in groups.values():
            if members & seed:
                out |= members
    return out


def _pin_counts(
    case: FloorSetCase,
    frame: PuzzleFrame,
    *,
    rows: int,
    cols: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for pin in case.pins_pos.tolist():
        counts[point_region(float(pin[0]), float(pin[1]), frame, rows=rows, cols=cols)] += 1
    return counts


def _net_demand(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    rows: int,
    cols: int,
) -> dict[str, float]:
    demand: dict[str, float] = defaultdict(float)
    for src, dst, weight in case.b2b_edges.tolist():
        for block_id in (int(src), int(dst)):
            if block_id in placements:
                demand[
                    point_region(*_center(placements[block_id]), frame, rows=rows, cols=cols)
                ] += abs(float(weight)) * 0.5
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        block = int(block_idx)
        pin = int(pin_idx)
        if block in placements:
            demand[point_region(*_center(placements[block]), frame, rows=rows, cols=cols)] += abs(
                float(weight)
            )
        if 0 <= pin < len(case.pins_pos):
            x, y = [float(value) for value in case.pins_pos[pin].tolist()]
            demand[point_region(x, y, frame, rows=rows, cols=cols)] += abs(float(weight))
    return demand


def _repair_reachability(
    occupancy_ratio: float,
    mib_group_ratio: float,
    slack: float,
    area: float,
) -> float:
    slack_score = min(max(slack / max(area, 1e-9), 0.0), 1.0)
    crowding_penalty = min(max(occupancy_ratio - 0.85, 0.0), 1.0)
    macro_penalty = min(mib_group_ratio, 1.0) * 0.25
    return max(slack_score - crowding_penalty - macro_penalty, 0.0)


def _map_summary(regions: list[dict[str, Any]]) -> dict[str, float]:
    if not regions:
        return {}
    return {
        "max_occupancy": max(float(row["occupancy_mask"]) for row in regions),
        "mean_free_space": _mean(float(row["free_space_mask"]) for row in regions),
        "max_pin_density": max(float(row["pin_density_heatmap"]) for row in regions),
        "max_repair_reachability": max(
            float(row["repair_reachability_mask"]) for row in regions
        ),
        "low_slack_region_count": sum(
            int(float(row["free_space_mask"]) < 0.10) for row in regions
        ),
    }


def _center(box: Placement) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(len(rows), 1)
