from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.diagnostics.region_topology import (
    block_region_assignment,
    box_center,
    free_space_fragmentation,
    point_region,
)
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame


def reconstructed_candidate_order(
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> list[int]:
    """Best-effort order proxy when Step6G construction trace is not persisted."""

    return sorted(
        placements,
        key=lambda idx: (
            round((box_center(placements[idx])[1] - frame.ymin) / max(frame.height, 1e-9), 4),
            round((box_center(placements[idx])[0] - frame.xmin) / max(frame.width, 1e-9), 4),
            idx,
        ),
    )


def candidate_ordering_trace(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    assignment: dict[str, Any],
    *,
    first_k: int = 16,
) -> dict[str, Any]:
    order = reconstructed_candidate_order(placements, frame)
    assignment_by_block = {int(row["block_id"]): row for row in assignment["assignments"]}
    mismatch_event = next(
        (
            _order_event(rank, block_id, assignment_by_block[block_id])
            for rank, block_id in enumerate(order)
            if int(assignment_by_block[block_id]["mismatch_grid_distance"]) >= 2
        ),
        None,
    )
    fragmentation_event = first_fragmentation_event(case, placements, frame, order)
    macro_event = first_macro_split_event(case, placements, frame, order)
    return {
        "case_id": case.case_id,
        "trace_confidence": "reconstructed",
        "confidence_reason": (
            "Step6G exact placement-order trace is not persisted; "
            "row-major center order is used as proxy."
        ),
        "first_k": [
            _order_event(rank, block_id, assignment_by_block[block_id])
            for rank, block_id in enumerate(order[:first_k])
        ],
        "first_major_region_mismatch": mismatch_event,
        "first_large_hole_fragmentation": fragmentation_event,
        "first_macro_member_away": macro_event,
    }


def _order_event(rank: int, block_id: int, assignment_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "block_id": block_id,
        "actual_region": assignment_row["actual_region"],
        "expected_region": assignment_row["expected_region"],
        "mismatch_grid_distance": assignment_row["mismatch_grid_distance"],
        "is_mib": assignment_row["is_mib"],
        "is_grouping": assignment_row["is_grouping"],
        "is_boundary": assignment_row["is_boundary"],
    }


def first_fragmentation_event(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    order: list[int],
) -> dict[str, Any] | None:
    prefix: dict[int, Placement] = {}
    for rank, block_id in enumerate(order):
        prefix[block_id] = placements[block_id]
        if len(prefix) < max(8, case.block_count // 8):
            continue
        frag = free_space_fragmentation(case, prefix, frame)
        if (
            float(frag["fragmentation_score"]) >= 1.0
            and int(frag["empty_component_count"]) >= 3
        ):
            return {
                "rank": rank,
                "block_id": block_id,
                "fragmentation_score": frag["fragmentation_score"],
                "empty_component_count": frag["empty_component_count"],
                "largest_empty_component_cell_fraction": frag[
                    "largest_empty_component_cell_fraction"
                ],
            }
    return None


def first_macro_split_event(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    order: list[int],
) -> dict[str, Any] | None:
    seen_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for rank, block_id in enumerate(order):
        for label, column in (
            ("MIB", ConstraintColumns.MIB),
            ("group", ConstraintColumns.CLUSTER),
        ):
            group_id = int(float(case.constraints[block_id, column].item()))
            if group_id <= 0:
                continue
            region = point_region(*box_center(placements[block_id]), frame)
            key = (label, group_id)
            seen_by_key[key].append(region)
            majority_region, majority_count = Counter(seen_by_key[key]).most_common(1)[0]
            if region != majority_region and majority_count >= 2:
                return {
                    "rank": rank,
                    "block_id": block_id,
                    "macro_kind": label,
                    "macro_id": group_id,
                    "actual_region": region,
                    "majority_region": majority_region,
                    "seen_region_count": len(set(seen_by_key[key])),
                }
    return None


def trace_from_layout(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    clusters: dict[str, Any],
) -> dict[str, Any]:
    assignment = block_region_assignment(case, placements, frame, clusters)
    return candidate_ordering_trace(case, placements, frame, assignment)
