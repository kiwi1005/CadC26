from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase

from .boxes import pairwise_intersection_area
from .constraints import HardLegalitySummary


def positions_from_case_targets(case: FloorSetCase) -> list[tuple[float, float, float, float]]:
    if case.target_positions is None:
        raise ValueError("case.target_positions is required for target-position legality checks")
    return [tuple(float(v) for v in row.tolist()) for row in case.target_positions]


def check_non_overlap(positions: list[tuple[float, float, float, float]], tolerance: float = 1e-9) -> int:
    violations = 0
    as_tensors = [torch.tensor(p, dtype=torch.float32) for p in positions]
    for i in range(len(as_tensors)):
        for j in range(i + 1, len(as_tensors)):
            if pairwise_intersection_area(as_tensors[i], as_tensors[j]) > tolerance:
                violations += 1
    return violations


def check_area_tolerance(
    positions: list[tuple[float, float, float, float]],
    target_areas: torch.Tensor,
    *,
    tolerance: float = 0.01,
    skip_indices: set[int] | None = None,
) -> int:
    skip_indices = skip_indices or set()
    violations = 0
    for idx, ((_, _, w, h), target_area) in enumerate(zip(positions, target_areas.tolist(), strict=False)):
        if idx in skip_indices:
            continue
        if target_area <= 0:
            continue
        if abs((w * h) - target_area) / float(target_area) > tolerance:
            violations += 1
    return violations


def check_dimension_hard_constraints(
    positions: list[tuple[float, float, float, float]],
    target_positions: torch.Tensor | None,
    constraints: torch.Tensor | None,
    *,
    tolerance: float = 1e-4,
) -> int:
    if target_positions is None or constraints is None:
        return 0
    violations = 0
    for idx, pos in enumerate(positions):
        fixed = bool(constraints[idx, ConstraintColumns.FIXED].item())
        preplaced = bool(constraints[idx, ConstraintColumns.PREPLACED].item())
        if not (fixed or preplaced):
            continue
        tx, ty, tw, th = [float(v) for v in target_positions[idx].tolist()]
        px, py, pw, ph = pos
        if abs(pw - tw) > tolerance or abs(ph - th) > tolerance:
            violations += 1
            continue
        if preplaced and (abs(px - tx) > tolerance or abs(py - ty) > tolerance):
            violations += 1
    return violations


def summarize_hard_legality(case: FloorSetCase, positions: list[tuple[float, float, float, float]]) -> HardLegalitySummary:
    fixed_or_preplaced = {
        idx
        for idx in range(case.block_count)
        if bool(case.constraints[idx, ConstraintColumns.FIXED].item())
        or bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    }
    overlap_violations = check_non_overlap(positions)
    area_violations = check_area_tolerance(positions, case.area_targets, skip_indices=fixed_or_preplaced)
    dimension_violations = check_dimension_hard_constraints(positions, case.target_positions, case.constraints)
    return HardLegalitySummary(
        is_feasible=(overlap_violations == 0 and area_violations == 0 and dimension_violations == 0),
        overlap_violations=overlap_violations,
        area_violations=area_violations,
        dimension_violations=dimension_violations,
    )
