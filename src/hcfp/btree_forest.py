"""Small anchor-aware B*-Tree forest challengers.

This module is intentionally a bounded experiment, not a replacement for the
existing candidate lane.  It treats every movable spatial island as a small
tree component and tries a few placements in the free strips of the current
bounding box.  The incumbent remains the caller's responsibility; this helper
only returns candidates accepted by the exact feasibility verifier.
"""

from __future__ import annotations

from typing import Any

import torch

from hcfp.btree import BStarTree
from hcfp.island_relocation import detect_islands, generate_island_relocations
from hcfp.verify import OVERLAP_EPS, as_xywh, bbox, verify_feasible


Tensor = torch.Tensor


def btree_forest_candidates(
    case: Any,
    placements: Any,
    tree: BStarTree | None = None,
    *,
    max_candidates: int = 8,
    proximity: float = 0.0,
) -> tuple[Tensor, ...]:
    """Return a bounded set of exact-feasible forest challengers.

    ``placements`` is the current ``[N,4]`` layout.  Fixed and preplaced
    blocks are anchors: they are copied byte-for-byte from the input and are
    never part of a moved island.  Movable islands are either rigidly moved by
    the existing relocation helper or repacked by a two-orientation local
    B*-Tree inside a current free rectangle.  A candidate is emitted only
    after :func:`hcfp.verify.verify_feasible` accepts it.

    The optional ``tree`` is a global B*-Tree.  Its depth-first order is used
    as a stable ordering hint for local forest components; disconnected local
    subtrees fall back to a deterministic index order.
    """

    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
        raise ValueError("max_candidates must be an integer")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    if max_candidates == 0:
        return ()
    if not torch.isfinite(torch.as_tensor(float(proximity))):
        raise ValueError("proximity must be finite")
    if float(proximity) < 0.0:
        raise ValueError("proximity must be non-negative")

    source = as_xywh(placements)
    n = int(source.shape[0])
    expected = getattr(case, "n", n)
    if int(expected) != n:
        raise ValueError("placements must match case.n")

    protected = _protected_mask(case, n)
    islands = detect_islands(source, proximity=float(proximity))
    core = max(
        islands, key=lambda members: (sum(_area(source[i]) for i in members), members)
    )
    order = _tree_order(tree, n)
    ordered_islands = tuple(
        sorted(
            islands,
            key=lambda members: (
                members == core,
                min(order.index(member) for member in members),
                members,
            ),
        )
    )

    found: list[Tensor] = []
    seen: set[tuple[float, ...]] = set()

    # Rigid relocation is the cheapest forest move and already handles anchor
    # protection and collision checking.  Keep it first so this experiment
    # cannot hide a useful incumbent behind a local repack.
    for relocation in generate_island_relocations(
        case,
        source,
        components=ordered_islands,
        proximity=float(proximity),
        max_candidates=max_candidates,
        preplaced_mask=protected,
    ):
        _append_verified(case, relocation.placement, found, seen, max_candidates)
        if len(found) >= max_candidates:
            return tuple(found)

    bounds = bbox(source)
    for members in ordered_islands:
        if members == core or bool(protected[list(members)].any()):
            continue
        local_order = tuple(
            sorted(members, key=lambda member: (order.index(member), member))
        )
        dimensions = _exact_dimensions(case, source, local_order)
        local_trees = _local_trees(len(local_order))
        obstacles = tuple(index for index in range(n) if index not in members)
        regions = _free_regions(bounds, source, obstacles)
        for orientation, local_tree in local_trees:
            packed = local_tree.pack(dimensions)
            width = float((packed[:, 0] + packed[:, 2]).max())
            height = float((packed[:, 1] + packed[:, 3]).max())
            for region in regions:
                left, bottom, right, top = region
                if (
                    width > right - left + OVERLAP_EPS
                    or height > top - bottom + OVERLAP_EPS
                ):
                    continue
                candidate = source.clone()
                shift = (
                    candidate.new_tensor((left, bottom))
                    - packed[:, :2].min(dim=0).values
                )
                candidate[list(local_order), :2] = packed[:, :2].to(candidate) + shift
                candidate[list(local_order), 2:4] = packed[:, 2:4].to(candidate)
                candidate[protected] = source[protected]
                _append_verified(case, candidate, found, seen, max_candidates)
                if len(found) >= max_candidates:
                    return tuple(found)
    return tuple(found)


# Short alias for callers that prefer the noun used in experiment notes.
forest_candidates = btree_forest_candidates


def _protected_mask(case: Any, n: int) -> Tensor:
    fixed = torch.as_tensor(
        getattr(case, "fixed_mask", torch.zeros(n)), dtype=torch.bool
    ).reshape(-1)
    preplaced = torch.as_tensor(
        getattr(case, "preplaced_mask", torch.zeros(n)), dtype=torch.bool
    ).reshape(-1)
    if fixed.numel() != n or preplaced.numel() != n:
        raise ValueError("case masks must have shape [N]")
    return fixed | preplaced


def _tree_order(tree: BStarTree | None, n: int) -> list[int]:
    if tree is None:
        return list(range(n))
    if tree.block_count != n:
        raise ValueError("tree.block_count must match case.n")
    order: list[int] = []

    def visit(node: int) -> None:
        order.append(node)
        if tree.left[node] >= 0:
            visit(tree.left[node])
        if tree.right[node] >= 0:
            visit(tree.right[node])

    visit(tree.root)
    return order


def _local_trees(count: int) -> tuple[tuple[str, BStarTree], ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (
            ("single", BStarTree.from_edges(torch.empty((0, 3), dtype=torch.long), 1)),
        )
    horizontal = torch.tensor(
        tuple((index, index + 1, 0) for index in range(count - 1)), dtype=torch.long
    )
    vertical = torch.tensor(
        tuple((index, index + 1, 1) for index in range(count - 1)), dtype=torch.long
    )
    return (
        ("horizontal", BStarTree.from_edges(horizontal, count)),
        ("vertical", BStarTree.from_edges(vertical, count)),
    )


def _exact_dimensions(case: Any, source: Tensor, members: tuple[int, ...]) -> Tensor:
    dimensions = source[list(members), 2:4].clone()
    areas = getattr(case, "area", None)
    if areas is None:
        return dimensions
    target = torch.as_tensor(areas, dtype=source.dtype, device=source.device).reshape(
        -1
    )
    if target.numel() != source.shape[0]:
        raise ValueError("case.area must have shape [N]")
    protected = _protected_mask(case, int(source.shape[0]))
    for local, member in enumerate(members):
        if bool(protected[member]):
            continue
        ratio = float(dimensions[local, 0] / dimensions[local, 1])
        if ratio <= 0.0:
            continue
        width = torch.sqrt(target[member] * dimensions.new_tensor(ratio))
        dimensions[local, 0] = width
        dimensions[local, 1] = target[member] / width
    return dimensions


def _free_regions(
    bounds: tuple[float, float, float, float],
    source: Tensor,
    obstacles: tuple[int, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    left, bottom, right, top = bounds
    regions: list[tuple[float, float, float, float]] = [(left, bottom, right, top)]
    for index in obstacles:
        ox, oy, ow, oh = (float(value) for value in source[index])
        regions.extend(
            (
                (left, bottom, min(right, ox), top),
                (max(left, ox + ow), bottom, right, top),
                (left, bottom, right, min(top, oy)),
                (left, max(bottom, oy + oh), right, top),
            )
        )
    valid = [
        region
        for region in regions
        if region[2] - region[0] > OVERLAP_EPS and region[3] - region[1] > OVERLAP_EPS
    ]
    return tuple(
        sorted(
            set(valid),
            key=lambda region: (
                -(region[2] - region[0]) * (region[3] - region[1]),
                region,
            ),
        )
    )


def _append_verified(
    case: Any,
    candidate: Tensor,
    found: list[Tensor],
    seen: set[tuple[float, ...]],
    limit: int,
) -> None:
    if len(found) >= limit or not verify_feasible(case, candidate):
        return
    key = tuple(round(float(value), 10) for value in candidate.reshape(-1))
    if key in seen:
        return
    seen.add(key)
    found.append(candidate.clone())


def _area(box: Tensor) -> float:
    return float(box[2] * box[3])


__all__ = ["btree_forest_candidates", "forest_candidates"]
