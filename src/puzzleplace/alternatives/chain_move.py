"""Step7R chain-move operator helpers.

Phase 1 implements the k=2 chain as a center swap.  The operator is intentionally
sidecar-only: it validates candidate geometry and emits compact JSON-compatible
candidate records, but it does not mutate layouts or touch contest runtime paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

EPS = 1e-9
Box = tuple[float, float, float, float]
Center = tuple[float, float]

__all__ = [
    "propose_swap_candidates",
    "swap_overlaps_neighbors",
    "swap_violates_boundary",
    "swap_violates_mib",
    "validate_swap",
]


def propose_swap_candidates(
    case_layout: Mapping[str, Any],
    *,
    seed_pairs: Iterable[Sequence[int | str]],
    fixed_block_ids: Iterable[int | str],
) -> list[dict[str, Any]]:
    """Validate deterministic k=2 swap candidates for the requested seed pairs."""

    fixed_tokens = {_id_token(block_id) for block_id in fixed_block_ids}
    return [
        validate_swap(case_layout, seed_pair, fixed_block_ids=fixed_tokens)
        for seed_pair in seed_pairs
    ]


def validate_swap(
    case_layout: Mapping[str, Any],
    swap_pair: Sequence[int | str],
    *,
    fixed_block_ids: Iterable[int | str] = (),
) -> dict[str, Any]:
    """Return a ``step7r_swap_candidate_v1`` validation record for one pair."""

    fixed_tokens = {_id_token(block_id) for block_id in fixed_block_ids}
    pair_ids = list(swap_pair[:2])
    block_by_id = _block_map(case_layout)
    if len(pair_ids) != 2 or _id_token(pair_ids[0]) == _id_token(pair_ids[1]):
        return _candidate_record(
            case_layout,
            pair_ids,
            {},
            legal=False,
            mib_guard_pass=False,
            boundary_guard_pass=False,
            rejection_reason="invalid_pair",
        )
    block_a = block_by_id.get(_id_token(pair_ids[0]))
    block_b = block_by_id.get(_id_token(pair_ids[1]))
    if block_a is None or block_b is None:
        return _candidate_record(
            case_layout,
            pair_ids,
            {},
            legal=False,
            mib_guard_pass=False,
            boundary_guard_pass=False,
            rejection_reason="missing_block",
        )

    post_centers = {
        _id_token(block_a["block_id"]): list(_center(block_b)),
        _id_token(block_b["block_id"]): list(_center(block_a)),
    }
    fixed = _is_fixed(block_a, fixed_tokens) or _is_fixed(block_b, fixed_tokens)
    overlaps = swap_overlaps_neighbors(case_layout, block_a["block_id"], block_b["block_id"])
    mib_violation = swap_violates_mib(case_layout, block_a["block_id"], block_b["block_id"])
    boundary_violation = swap_violates_boundary(
        case_layout, block_a["block_id"], block_b["block_id"]
    )
    rejection_reason: str | None = None
    if fixed:
        rejection_reason = "fixed_block"
    elif overlaps:
        rejection_reason = "neighbor_overlap"
    elif mib_violation:
        rejection_reason = "mib_violation"
    elif boundary_violation:
        rejection_reason = "boundary_violation"
    return _candidate_record(
        case_layout,
        [block_a["block_id"], block_b["block_id"]],
        post_centers,
        legal=rejection_reason is None,
        mib_guard_pass=not mib_violation,
        boundary_guard_pass=not boundary_violation,
        rejection_reason=rejection_reason,
    )


def swap_overlaps_neighbors(
    case_layout: Mapping[str, Any], block_id_a: int | str, block_id_b: int | str
) -> bool:
    """Return true if either swapped block overlaps any other post-swap block."""

    after = _after_swap_boxes(case_layout, block_id_a, block_id_b)
    token_a = _id_token(block_id_a)
    token_b = _id_token(block_id_b)
    for moved_token in (token_a, token_b):
        moved = after.get(moved_token)
        if moved is None:
            return True
        for other_token, other in after.items():
            if other_token != moved_token and _overlap_area(moved, other) > EPS:
                return True
    return False


def swap_violates_mib(
    case_layout: Mapping[str, Any], block_id_a: int | str, block_id_b: int | str
) -> bool:
    """Return true when a swap crosses an MIB equal-shape group boundary."""

    group_by_block = _mib_group_by_block(case_layout)
    group_a = group_by_block.get(_id_token(block_id_a))
    group_b = group_by_block.get(_id_token(block_id_b))
    if group_a is None and group_b is None:
        return False
    return group_a != group_b


def swap_violates_boundary(
    case_layout: Mapping[str, Any], block_id_a: int | str, block_id_b: int | str
) -> bool:
    """Return true if either swapped block falls outside the case boundary."""

    boundary = _boundary_box(case_layout.get("boundary"))
    if boundary is None:
        return False
    after = _after_swap_boxes(case_layout, block_id_a, block_id_b)
    for token in (_id_token(block_id_a), _id_token(block_id_b)):
        box = after.get(token)
        if box is None or not _box_inside(box, boundary):
            return True
    return False


def _candidate_record(
    case_layout: Mapping[str, Any],
    swap_pair: Sequence[Any],
    post_swap_centers: Mapping[str, list[float]],
    *,
    legal: bool,
    mib_guard_pass: bool,
    boundary_guard_pass: bool,
    rejection_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema": "step7r_swap_candidate_v1",
        "case_id": case_layout.get("case_id"),
        "swap_pair": list(swap_pair),
        "post_swap_centers": dict(post_swap_centers),
        "legal": legal,
        "area_preserving": True,
        "mib_guard_pass": mib_guard_pass,
        "boundary_guard_pass": boundary_guard_pass,
        "rejection_reason": rejection_reason,
    }


def _block_map(case_layout: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    blocks = case_layout.get("blocks", [])
    if not isinstance(blocks, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for fallback_id, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_id = block.get("block_id", fallback_id)
        mapped[_id_token(block_id)] = {**block, "block_id": block_id}
    return mapped


def _after_swap_boxes(
    case_layout: Mapping[str, Any], block_id_a: int | str, block_id_b: int | str
) -> dict[str, Box]:
    block_by_id = _block_map(case_layout)
    token_a = _id_token(block_id_a)
    token_b = _id_token(block_id_b)
    block_a = block_by_id.get(token_a)
    block_b = block_by_id.get(token_b)
    if block_a is None or block_b is None:
        return {}
    center_a = _center(block_a)
    center_b = _center(block_b)
    boxes: dict[str, Box] = {}
    for token, block in block_by_id.items():
        if token == token_a:
            boxes[token] = _box_at_center(block, center_b)
        elif token == token_b:
            boxes[token] = _box_at_center(block, center_a)
        else:
            boxes[token] = _box(block)
    return boxes


def _mib_group_by_block(case_layout: Mapping[str, Any]) -> dict[str, str]:
    group_by_block: dict[str, str] = {}
    groups = case_layout.get("mib_groups")
    if isinstance(groups, Mapping):
        for group_id, members in groups.items():
            for member in _group_members(members):
                group_by_block[_id_token(member)] = str(group_id)
    elif isinstance(groups, list):
        for index, group in enumerate(groups):
            if isinstance(group, Mapping):
                group_id = group.get("group_id", group.get("mib_id", group.get("id", index)))
                members = (
                    group.get("block_ids")
                    or group.get("members")
                    or group.get("blocks")
                    or []
                )
            else:
                group_id = index
                members = group
            for member in _group_members(members):
                group_by_block[_id_token(member)] = str(group_id)

    for block in _block_map(case_layout).values():
        mib_id = block.get("mib", block.get("mib_group"))
        if mib_id not in (None, "", 0, "0"):
            group_by_block.setdefault(_id_token(block["block_id"]), str(mib_id))
    return group_by_block


def _group_members(members: Any) -> list[Any]:
    if not isinstance(members, list | tuple | set):
        return []
    result: list[Any] = []
    for member in members:
        if isinstance(member, Mapping):
            result.append(member.get("block_id"))
        else:
            result.append(member)
    return [member for member in result if member is not None]


def _boundary_box(boundary: Any) -> Box | None:
    if boundary is None:
        return None
    if isinstance(boundary, Mapping):
        x0 = _float(boundary.get("x", boundary.get("min_x", boundary.get("left", 0.0))))
        y0 = _float(boundary.get("y", boundary.get("min_y", boundary.get("bottom", 0.0))))
        if "w" in boundary or "width" in boundary:
            x1 = x0 + _float(boundary.get("w", boundary.get("width", 0.0)))
        else:
            x1 = _float(boundary.get("max_x", boundary.get("right", x0)))
        if "h" in boundary or "height" in boundary:
            y1 = y0 + _float(boundary.get("h", boundary.get("height", 0.0)))
        else:
            y1 = _float(boundary.get("max_y", boundary.get("top", y0)))
        return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))
    if isinstance(boundary, list | tuple) and len(boundary) == 4:
        x, y, w, h = boundary
        return (_float(x), _float(y), _float(w), _float(h))
    return None


def _box_inside(box: Box, boundary: Box) -> bool:
    x, y, w, h = box
    bx, by, bw, bh = boundary
    return x >= bx - EPS and y >= by - EPS and x + w <= bx + bw + EPS and y + h <= by + bh + EPS


def _is_fixed(block: Mapping[str, Any], fixed_tokens: set[str]) -> bool:
    return (
        _id_token(block.get("block_id")) in fixed_tokens
        or bool(block.get("fixed"))
        or bool(block.get("preplaced"))
    )


def _center(block: Mapping[str, Any]) -> Center:
    return (
        _float(block.get("x")) + _float(block.get("w")) / 2.0,
        _float(block.get("y")) + _float(block.get("h")) / 2.0,
    )


def _box(block: Mapping[str, Any]) -> Box:
    return (
        _float(block.get("x")),
        _float(block.get("y")),
        _float(block.get("w")),
        _float(block.get("h")),
    )


def _box_at_center(block: Mapping[str, Any], center: Center) -> Box:
    w = _float(block.get("w"))
    h = _float(block.get("h"))
    return (center[0] - w / 2.0, center[1] - h / 2.0, w, h)


def _overlap_area(a: Box, b: Box) -> float:
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _id_token(value: Any) -> str:
    return str(value)
