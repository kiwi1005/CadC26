from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.region_topology import box_center, point_region
from puzzleplace.diagnostics.repair_radius import (
    changed_blocks,
    frame_violation_blocks,
    overlap_blocks,
)
from puzzleplace.repair.overlap_resolver import resolve_overlaps
from puzzleplace.repair.shape_normalizer import normalize_shapes
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

RepairMode = Literal[
    "current_repair_baseline",
    "geometry_window_repair",
    "region_cell_repair",
    "graph_hop_repair",
    "macro_component_repair",
    "cascade_capped_repair",
    "rollback_to_original",
]


@dataclass(frozen=True, slots=True)
class BoundedRepairResult:
    placements: dict[int, Placement]
    repair_mode: RepairMode
    repair_seed: set[int]
    repair_region: set[int]
    expansion_reasons: dict[int, list[str]]
    repair_radius_exceeded: bool
    reject_reason: str | None
    runtime_estimate_ms: float


def bounded_repair(
    case: FloorSetCase,
    *,
    baseline: dict[int, Placement],
    candidate: dict[int, Placement],
    frame: PuzzleFrame,
    mode: RepairMode,
    max_moved_fraction: float = 0.35,
    max_affected_regions: int = 8,
) -> BoundedRepairResult:
    start = time.perf_counter()
    seed = repair_seed_blocks(case, baseline, candidate, frame)
    if mode in {"current_repair_baseline", "rollback_to_original"}:
        placements = dict(candidate if mode == "current_repair_baseline" else baseline)
        return BoundedRepairResult(
            placements=placements,
            repair_mode=mode,
            repair_seed=seed,
            repair_region=(
                set(range(case.block_count)) if mode == "current_repair_baseline" else set()
            ),
            expansion_reasons={idx: ["current_repair_baseline"] for idx in range(case.block_count)}
            if mode == "current_repair_baseline"
            else {},
            repair_radius_exceeded=False,
            reject_reason=None,
            runtime_estimate_ms=(time.perf_counter() - start) * 1000.0,
        )
    repair_region, reasons = expand_repair_region(
        case,
        baseline=baseline,
        candidate=candidate,
        frame=frame,
        seed=seed,
        mode=mode,
    )
    radius_exceeded = (
        len(repair_region) / max(case.block_count, 1) > max_moved_fraction
        or affected_region_count_for_blocks(repair_region, baseline, frame) > max_affected_regions
    )
    if radius_exceeded and mode == "cascade_capped_repair":
        return BoundedRepairResult(
            placements=dict(baseline),
            repair_mode=mode,
            repair_seed=seed,
            repair_region=repair_region,
            expansion_reasons=reasons,
            repair_radius_exceeded=True,
            reject_reason="repair_radius_exceeded",
            runtime_estimate_ms=(time.perf_counter() - start) * 1000.0,
        )
    repaired = apply_local_repair(case, baseline, candidate, repair_region)
    return BoundedRepairResult(
        placements=repaired,
        repair_mode=mode,
        repair_seed=seed,
        repair_region=repair_region,
        expansion_reasons=reasons,
        repair_radius_exceeded=radius_exceeded,
        reject_reason="repair_radius_exceeded" if radius_exceeded else None,
        runtime_estimate_ms=(time.perf_counter() - start) * 1000.0,
    )


def repair_seed_blocks(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    candidate: dict[int, Placement],
    frame: PuzzleFrame,
) -> set[int]:
    seed = changed_blocks(baseline, candidate)
    seed |= overlap_blocks(candidate)
    seed |= frame_violation_blocks(candidate, frame)
    seed |= macro_violation_blocks(case, candidate)
    seed |= fixed_preplaced_violation_blocks(case, candidate)
    return seed


def expand_repair_region(
    case: FloorSetCase,
    *,
    baseline: dict[int, Placement],
    candidate: dict[int, Placement],
    frame: PuzzleFrame,
    seed: set[int],
    mode: RepairMode,
) -> tuple[set[int], dict[int, list[str]]]:
    region = set(seed)
    reasons: dict[int, list[str]] = {idx: ["seed"] for idx in seed}
    if mode == "geometry_window_repair":
        for idx in geometry_window_blocks(baseline, candidate, seed):
            region.add(idx)
            reasons.setdefault(idx, []).append("geometry_window")
    elif mode == "region_cell_repair":
        seed_regions = {
            point_region(*box_center(candidate.get(idx, baseline[idx])), frame)
            for idx in seed
            if idx in baseline
        }
        for idx, box in baseline.items():
            if point_region(*box_center(box), frame) in seed_regions:
                region.add(idx)
                reasons.setdefault(idx, []).append("same_region_cell")
    elif mode == "graph_hop_repair":
        for idx in graph_hop_blocks(case, seed, hops=1):
            region.add(idx)
            reasons.setdefault(idx, []).append("graph_hop")
    elif mode == "macro_component_repair":
        for idx in macro_component_blocks(case, seed):
            region.add(idx)
            reasons.setdefault(idx, []).append("macro_component")
    elif mode == "cascade_capped_repair":
        for idx in graph_hop_blocks(case, seed, hops=1) | macro_component_blocks(case, seed):
            region.add(idx)
            reasons.setdefault(idx, []).append("cascade_graph_or_macro")
    return region, reasons


def apply_local_repair(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    candidate: dict[int, Placement],
    repair_region: set[int],
) -> dict[int, Placement]:
    merged = dict(baseline)
    for idx in repair_region:
        if idx in candidate:
            merged[idx] = candidate[idx]
    normalized = normalize_shapes(case, merged)
    preplaced = {
        idx
        for idx in range(case.block_count)
        if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    }
    locked = set(range(case.block_count)) - repair_region
    locked |= preplaced
    resolved, _moved = resolve_overlaps(normalized, locked_blocks=locked)
    return {idx: resolved[idx] for idx in range(case.block_count)}


def geometry_window_blocks(
    baseline: dict[int, Placement],
    candidate: dict[int, Placement],
    seed: set[int],
    *,
    expansion: float = 0.25,
) -> set[int]:
    seed_boxes = [candidate.get(idx, baseline[idx]) for idx in seed if idx in baseline]
    if not seed_boxes:
        return set()
    xmin = min(box[0] for box in seed_boxes)
    ymin = min(box[1] for box in seed_boxes)
    xmax = max(box[0] + box[2] for box in seed_boxes)
    ymax = max(box[1] + box[3] for box in seed_boxes)
    pad = max(xmax - xmin, ymax - ymin) * expansion
    window = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    return {
        idx
        for idx, box in baseline.items()
        if not (
            box[0] + box[2] < window[0]
            or box[0] > window[2]
            or box[1] + box[3] < window[1]
            or box[1] > window[3]
        )
    }


def graph_hop_blocks(case: FloorSetCase, seed: set[int], *, hops: int = 1) -> set[int]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(case.block_count)}
    for src, dst, _weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i >= 0 and j >= 0:
            adjacency[i].add(j)
            adjacency[j].add(i)
    visited = set(seed)
    queue: deque[tuple[int, int]] = deque((idx, 0) for idx in seed)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return visited


def macro_component_blocks(case: FloorSetCase, seed: set[int]) -> set[int]:
    groups: dict[tuple[ConstraintColumns, int], set[int]] = defaultdict(set)
    for idx in range(case.block_count):
        for column in (
            ConstraintColumns.MIB,
            ConstraintColumns.CLUSTER,
            ConstraintColumns.BOUNDARY,
        ):
            value = int(float(case.constraints[idx, column].item()))
            if value > 0:
                groups[(column, value)].add(idx)
    out = set(seed)
    for members in groups.values():
        if members & seed:
            out |= members
    return out


def macro_violation_blocks(case: FloorSetCase, positions: dict[int, Placement]) -> set[int]:
    out: set[int] = set()
    for column in (ConstraintColumns.MIB, ConstraintColumns.CLUSTER):
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(case.block_count):
            value = int(float(case.constraints[idx, column].item()))
            if value > 0:
                groups[value].append(idx)
        for members in groups.values():
            regions = {
                (
                    round(positions[idx][0] / 10.0),
                    round(positions[idx][1] / 10.0),
                )
                for idx in members
                if idx in positions
            }
            if len(regions) > max(2, len(members) // 2):
                out |= set(members)
    return out


def fixed_preplaced_violation_blocks(
    case: FloorSetCase,
    positions: dict[int, Placement],
    *,
    eps: float = 1e-4,
) -> set[int]:
    if case.target_positions is None:
        return set()
    out: set[int] = set()
    for idx, box in positions.items():
        fixed = bool(case.constraints[idx, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
        if not (fixed or preplaced):
            continue
        tx, ty, tw, th = [float(v) for v in case.target_positions[idx].tolist()]
        if abs(box[2] - tw) > eps or abs(box[3] - th) > eps:
            out.add(idx)
        if preplaced and (abs(box[0] - tx) > eps or abs(box[1] - ty) > eps):
            out.add(idx)
    return out


def affected_region_count_for_blocks(
    blocks: set[int],
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> int:
    return len(
        {
            point_region(*box_center(placements[idx]), frame)
            for idx in blocks
            if idx in placements
        }
    )
