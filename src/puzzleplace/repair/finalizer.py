from __future__ import annotations

from dataclasses import dataclass

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.eval.violation import summarize_violation_profile
from puzzleplace.geometry import summarize_hard_legality

from .intent_preserver import measure_intent_preservation
from .overlap_resolver import resolve_overlaps
from .shape_normalizer import normalize_shapes
from .shelf_packer import shelf_pack_missing


@dataclass(slots=True)
class RepairReport:
    hard_feasible_before: bool
    hard_feasible_after: bool
    overlap_pairs_before: int
    overlap_pairs_after: int
    total_overlap_area_before: float
    total_overlap_area_after: float
    area_violations_before: int
    area_violations_after: int
    dimension_violations_before: int
    dimension_violations_after: int
    mean_displacement: float
    max_displacement: float
    preserved_x_order_fraction: float
    shelf_fallback_count: int
    moved_block_count: int


@dataclass(slots=True)
class RepairResult:
    positions: list[tuple[float, float, float, float]]
    report: RepairReport


def finalize_layout(
    case: FloorSetCase,
    proposed_positions: dict[int, tuple[float, float, float, float]],
) -> RepairResult:
    before_profile = summarize_violation_profile(case, proposed_positions)
    before_hard = None
    if len(proposed_positions) == case.block_count:
        before_hard = summarize_hard_legality(
            case,
            [proposed_positions[idx] for idx in range(case.block_count)],
        )

    locked_blocks = {
        idx
        for idx in range(case.block_count)
        if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    }
    normalized = normalize_shapes(case, proposed_positions)
    resolved, moved = resolve_overlaps(normalized, locked_blocks=locked_blocks)
    missing_blocks = [idx for idx in range(case.block_count) if idx not in resolved]
    packed, shelf_count = shelf_pack_missing(case, resolved, missing_blocks)
    repaired, moved2 = resolve_overlaps(packed, locked_blocks=locked_blocks)
    moved |= moved2

    ordered = [repaired[idx] for idx in range(case.block_count)]
    after_hard = summarize_hard_legality(case, ordered)
    after_profile = summarize_violation_profile(case, repaired)
    intent_stats = measure_intent_preservation(proposed_positions, repaired)
    report = RepairReport(
        hard_feasible_before=before_hard.is_feasible if before_hard is not None else False,
        hard_feasible_after=after_hard.is_feasible,
        overlap_pairs_before=before_profile.overlap_pairs,
        overlap_pairs_after=after_profile.overlap_pairs,
        total_overlap_area_before=before_profile.total_overlap_area,
        total_overlap_area_after=after_profile.total_overlap_area,
        area_violations_before=before_profile.area_violations,
        area_violations_after=after_profile.area_violations,
        dimension_violations_before=before_profile.dimension_violations,
        dimension_violations_after=after_profile.dimension_violations,
        mean_displacement=float(intent_stats["mean_displacement"]),
        max_displacement=float(intent_stats["max_displacement"]),
        preserved_x_order_fraction=float(intent_stats["preserved_x_order_fraction"]),
        shelf_fallback_count=shelf_count,
        moved_block_count=len(moved),
    )
    return RepairResult(positions=ordered, report=report)
