from __future__ import annotations

from dataclasses import dataclass

import torch

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.geometry.boxes import pairwise_intersection_area


@dataclass(slots=True)
class ViolationProfile:
    block_count: int
    placed_count: int
    semantic_placed_fraction: float
    overlap_pairs: int
    total_overlap_area: float
    area_violations: int
    dimension_violations: int
    boundary_distance: float
    connectivity_proxy_cost: float


def _area_violations(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> int:
    violations = 0
    for block_index, (_x, _y, width, height) in positions.items():
        target_area = float(case.area_targets[block_index].item())
        if target_area <= 0:
            continue
        if abs((width * height) - target_area) / max(target_area, 1e-6) > 0.05:
            violations += 1
    return violations


def _dimension_violations(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> int:
    if case.target_positions is None:
        return 0
    violations = 0
    for block_index, (x, y, width, height) in positions.items():
        fixed = bool(case.constraints[block_index, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
        if not (fixed or preplaced):
            continue
        tx, ty, tw, th = [float(v) for v in case.target_positions[block_index].tolist()]
        if tw > 0 and th > 0 and (abs(width - tw) > 1e-4 or abs(height - th) > 1e-4):
            violations += 1
            continue
        if preplaced and tx >= 0 and ty >= 0 and (abs(x - tx) > 1e-4 or abs(y - ty) > 1e-4):
            violations += 1
    return violations


def _boundary_distance(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> float:
    if not positions:
        return 0.0
    max_x = max(x + width for x, _y, width, _h in positions.values())
    max_y = max(y + height for _x, y, _w, height in positions.values())
    total = 0.0
    for block_index, (x, y, width, height) in positions.items():
        code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
        if code == 0:
            continue
        if code == 1:
            total += abs(x)
        elif code == 2:
            total += abs(max_x - (x + width))
        elif code == 4:
            total += abs(max_y - (y + height))
        elif code == 8:
            total += abs(y)
    return float(total)


def _connectivity_proxy_cost(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> float:
    total = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        if src == -1 or dst == -1:
            continue
        src_idx = int(src)
        dst_idx = int(dst)
        if src_idx not in positions or dst_idx not in positions:
            continue
        x1, y1, w1, h1 = positions[src_idx]
        x2, y2, w2, h2 = positions[dst_idx]
        total += float(weight) * (
            abs((x1 + w1 / 2.0) - (x2 + w2 / 2.0)) + abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0))
        )
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        if pin_idx == -1 or block_idx == -1:
            continue
        pin = int(pin_idx)
        block = int(block_idx)
        if block not in positions or pin >= len(case.pins_pos):
            continue
        px, py = [float(v) for v in case.pins_pos[pin].tolist()]
        x, y, width, height = positions[block]
        total += float(weight) * (abs(px - (x + width / 2.0)) + abs(py - (y + height / 2.0)))
    return float(total)


def summarize_violation_profile(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
    *,
    placed_count_override: int | None = None,
) -> ViolationProfile:
    overlap_pairs = 0
    total_overlap_area = 0.0
    items = list(positions.items())
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            left_box = torch.tensor(items[left][1], dtype=torch.float32)
            right_box = torch.tensor(items[right][1], dtype=torch.float32)
            area = float(pairwise_intersection_area(left_box, right_box))
            if area > 1e-9:
                overlap_pairs += 1
                total_overlap_area += area

    placed_count = placed_count_override if placed_count_override is not None else len(positions)
    return ViolationProfile(
        block_count=case.block_count,
        placed_count=placed_count,
        semantic_placed_fraction=placed_count / max(case.block_count, 1),
        overlap_pairs=overlap_pairs,
        total_overlap_area=float(total_overlap_area),
        area_violations=_area_violations(case, positions),
        dimension_violations=_dimension_violations(case, positions),
        boundary_distance=_boundary_distance(case, positions),
        connectivity_proxy_cost=_connectivity_proxy_cost(case, positions),
    )
