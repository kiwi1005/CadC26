from __future__ import annotations

import math

from puzzleplace.data import ConstraintColumns, FloorSetCase


def normalize_shapes(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> dict[int, tuple[float, float, float, float]]:
    normalized: dict[int, tuple[float, float, float, float]] = {}
    for block_index, (x, y, width, height) in positions.items():
        target_area = float(case.area_targets[block_index].item())
        fixed = bool(case.constraints[block_index, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
        if case.target_positions is not None and (fixed or preplaced):
            tx, ty, tw, th = [float(v) for v in case.target_positions[block_index].tolist()]
            if preplaced and tx >= 0 and ty >= 0:
                normalized[block_index] = (tx, ty, tw, th)
            else:
                normalized[block_index] = (x, y, tw, th)
            continue

        if target_area <= 0:
            normalized[block_index] = (x, y, width, height)
            continue
        aspect = max(width / max(height, 1e-6), 1e-3)
        fixed_width = math.sqrt(target_area * aspect)
        fixed_height = target_area / max(fixed_width, 1e-6)
        normalized[block_index] = (x, y, fixed_width, fixed_height)
    return normalized
