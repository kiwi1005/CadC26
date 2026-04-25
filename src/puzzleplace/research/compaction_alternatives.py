from __future__ import annotations

from dataclasses import dataclass, field

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.research.boundary_failure_attribution import compact_left_bottom
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    _bbox_boundary_satisfied_edges,
    _bbox_from_placements,
    _boundary_edges,
)


@dataclass(slots=True)
class EdgeAwareCompactionResult:
    placements: dict[int, Placement]
    locked_satisfied_boundary_owner_ids: list[int] = field(default_factory=list)
    promoted_boundary_block_ids: list[int] = field(default_factory=list)
    demoted_regular_hull_owner_ids: list[int] = field(default_factory=list)
    rejected_promotion_block_ids: list[int] = field(default_factory=list)


def _block_code(case: FloorSetCase, block_id: int, column: ConstraintColumns) -> int:
    return int(float(case.constraints[block_id, column].item()))


def _is_fixed_or_preplaced(case: FloorSetCase, block_id: int) -> bool:
    return bool(_block_code(case, block_id, ConstraintColumns.FIXED)) or bool(
        _block_code(case, block_id, ConstraintColumns.PREPLACED)
    )


def _is_regular(case: FloorSetCase, block_id: int) -> bool:
    return not any(
        [
            bool(_block_code(case, block_id, ConstraintColumns.BOUNDARY)),
            bool(_block_code(case, block_id, ConstraintColumns.MIB)),
            bool(_block_code(case, block_id, ConstraintColumns.CLUSTER)),
            _is_fixed_or_preplaced(case, block_id),
        ]
    )


def _edge_owned(
    edge: str, box: Placement, bbox: tuple[float, float, float, float], eps: float = 1e-4
) -> bool:
    x, y, w, h = box
    xmin, ymin, xmax, ymax = bbox
    if edge == "left":
        return abs(x - xmin) <= eps
    if edge == "right":
        return abs((x + w) - xmax) <= eps
    if edge == "bottom":
        return abs(y - ymin) <= eps
    if edge == "top":
        return abs((y + h) - ymax) <= eps
    raise ValueError(f"unknown edge: {edge}")


def _align_box_to_edge(
    edge: str, box: Placement, bbox: tuple[float, float, float, float]
) -> Placement:
    x, y, w, h = box
    xmin, ymin, xmax, ymax = bbox
    if edge == "left":
        return xmin, y, w, h
    if edge == "right":
        return xmax - w, y, w, h
    if edge == "bottom":
        return x, ymin, w, h
    if edge == "top":
        return x, ymax - h, w, h
    raise ValueError(f"unknown edge: {edge}")


def _overlaps(
    placements: dict[int, Placement],
    block_id: int,
    candidate: Placement,
    *,
    sep: float = 1e-4,
) -> bool:
    x, y, w, h = candidate
    for other, box in placements.items():
        if other == block_id:
            continue
        ox, oy, ow, oh = box
        if max(x, ox) < min(x + w, ox + ow) - sep and max(y, oy) < min(y + h, oy + oh) - sep:
            return True
    return False


def _preserved_left_bottom_compact(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    frame: PuzzleFrame | None,
    locked_ids: set[int],
    passes: int,
) -> dict[int, Placement]:
    """Left/bottom compact only the unlocked subset.

    This mirrors the Step6K simple compactor but keeps already satisfied
    boundary owners fixed so alternatives can be rejected/accepted without
    destroying previous boundary wins.
    """

    compacted = dict(placements)
    for _ in range(max(passes, 1)):
        before = dict(compacted)
        moving = {
            idx: box
            for idx, box in compacted.items()
            if idx not in locked_ids and not _is_fixed_or_preplaced(case, idx)
        }
        frozen = {idx: box for idx, box in compacted.items() if idx not in moving}
        movable_only = compact_left_bottom(
            case,
            {**frozen, **moving},
            frame=frame,
            passes=1,
            preserve_fixed_preplaced=True,
        )
        for idx in moving:
            compacted[idx] = movable_only[idx]
        if compacted == before:
            break
    return compacted


def edge_aware_compaction(
    case: FloorSetCase,
    placements: dict[int, Placement],
    *,
    frame: PuzzleFrame | None = None,
    target_hull: PuzzleFrame | None = None,
    passes: int = 3,
) -> EdgeAwareCompactionResult:
    """Produce the Step6L-D edge-aware compaction alternative.

    The algorithm is intentionally conservative:
    - lock boundary owners that are already satisfied against the final bbox;
    - attempt direct edge promotion for unsatisfied boundary blocks when legal;
    - compact remaining movable blocks left/bottom without touching locks.
    """

    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return EdgeAwareCompactionResult(placements=dict(placements))
    target_box = (
        (target_hull.xmin, target_hull.ymin, target_hull.xmax, target_hull.ymax)
        if target_hull is not None
        else bbox
    )

    locked_ids: set[int] = set()
    for idx, box in placements.items():
        code = _block_code(case, idx, ConstraintColumns.BOUNDARY)
        if code == 0:
            continue
        sat, total = _bbox_boundary_satisfied_edges(code, box, bbox)
        if total > 0 and sat == total:
            locked_ids.add(idx)

    promoted: list[int] = []
    rejected: list[int] = []
    updated = dict(placements)
    for idx in sorted(updated):
        if idx in locked_ids or _is_fixed_or_preplaced(case, idx):
            continue
        code = _block_code(case, idx, ConstraintColumns.BOUNDARY)
        if code == 0:
            continue
        required = _boundary_edges(code)
        if not required:
            continue
        if all(_edge_owned(edge, updated[idx], bbox) for edge in required):
            continue
        accepted = False
        for edge in required:
            candidate = _align_box_to_edge(edge, updated[idx], target_box)
            if frame is not None and not frame.contains_box(candidate):
                continue
            if _overlaps(updated, idx, candidate):
                continue
            updated[idx] = candidate
            locked_ids.add(idx)
            promoted.append(idx)
            accepted = True
            break
        if not accepted:
            rejected.append(idx)

    compacted = _preserved_left_bottom_compact(
        case, updated, frame=frame, locked_ids=locked_ids, passes=passes
    )
    new_bbox = _bbox_from_placements(compacted.values())
    demoted: list[int] = []
    if new_bbox is not None:
        for idx, box in placements.items():
            if not _is_regular(case, idx):
                continue
            old_owned = any(
                _edge_owned(edge, box, bbox)
                for edge in ("left", "right", "bottom", "top")
            )
            new_owned = any(
                _edge_owned(edge, compacted[idx], new_bbox)
                for edge in ("left", "right", "bottom", "top")
            )
            if old_owned and not new_owned:
                demoted.append(idx)

    return EdgeAwareCompactionResult(
        placements=compacted,
        locked_satisfied_boundary_owner_ids=sorted(locked_ids),
        promoted_boundary_block_ids=sorted(set(promoted)),
        demoted_regular_hull_owner_ids=sorted(set(demoted)),
        rejected_promotion_block_ids=sorted(set(rejected)),
    )
