"""Bounded exact local MIB-shape repairs for experiment-side obligation search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.contact_patch import _bounds, _closed_patch, _intersects
from hcfp.treemap import _Item, _allocate_items, _free_rectangles, _place_items
from hcfp.verify import mib_violation, verify_feasible


Tensor = torch.Tensor
_EPS = 1.0e-9


@dataclass(frozen=True)
class MibPatchCandidate:
    """One exact-feasible local repack that reduces a MIB shape debt."""

    placement: Tensor
    group_index: int
    anchor_member: int
    target_member: int
    members: tuple[int, ...]
    target_shape: tuple[float, float]
    mib_before: int
    mib_after: int


def mib_anchor_patch_candidates(
    case: Any,
    placements: Any,
    *,
    verify_case: Any | None = None,
    patch_sizes: tuple[int, ...] = (4, 8, 12, 16),
    max_candidates: int = 16,
) -> tuple[MibPatchCandidate, ...]:
    """Repair one compatible MIB member inside a fixed local frame.

    Fixed and preplaced members remain byte-for-byte unchanged.  All other
    MIB members are treated as local obstacles, so a candidate changes exactly
    one member toward an anchor shape while re-slicing only ordinary blockers.
    The caller may compose these candidates in a common loop to broadcast a
    shape across a MIB group without a global re-pack.
    """

    source = torch.as_tensor(placements, dtype=torch.float64, device="cpu").clone()
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("placements must have shape [N,4]")
    if max_candidates <= 0:
        return ()
    n = int(source.shape[0])
    membership = _membership(case, n)
    if not membership.numel():
        return ()
    areas = _areas(case, source)
    fixed = _mask(case, "fixed_mask", n)
    preplaced = _mask(case, "preplaced_mask", n)
    hard = fixed | preplaced
    protected_mib = membership.any(dim=0)
    bits = _boundary_bits(case, n)
    sizes = tuple(sorted({int(size) for size in patch_sizes if int(size) >= 2}))
    if not sizes:
        return ()

    before = mib_violation(case, source)
    if before <= 0:
        return ()
    global_bounds = _bounds(source)
    centers = source[:, :2] + 0.5 * source[:, 2:4]
    found: list[MibPatchCandidate] = []
    seen: set[tuple[float, ...]] = set()

    for group_index, row in enumerate(membership):
        members = tuple(
            int(index)
            for index in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        )
        if len(members) < 2:
            continue
        for anchor, shape in _anchor_shapes(source, areas, hard, members):
            if not _area_compatible(areas[list(members)], shape):
                continue
            targets = [
                member
                for member in members
                if not bool(hard[member])
                and not _same_shape(source[member, 2:4], shape)
            ]
            targets.sort(
                key=lambda member: (
                    -_shape_distance(source[member, 2:4], shape),
                    member,
                )
            )
            for target in targets:
                expanded = _resized_bounds(source[target], shape)
                seed = {
                    target,
                    *(
                        index
                        for index, box in enumerate(source)
                        if index != target and _intersects(box, expanded)
                    ),
                }
                nearest = torch.argsort(
                    torch.abs(centers - centers[target]).sum(dim=1), stable=True
                ).tolist()
                for size in sizes:
                    members_patch = _closed_patch(
                        source,
                        {
                            *seed,
                            *nearest[: max(0, size - len(seed))],
                        },
                        limit=size,
                    )
                    if members_patch is None:
                        continue
                    for candidate in _repack_member(
                        source,
                        members_patch,
                        target,
                        shape,
                        protected=hard | (protected_mib & ~_one_hot(n, target)),
                        boundary_bits=bits,
                        global_bounds=global_bounds,
                    ):
                        if not verify_feasible(verify_case or case, candidate):
                            continue
                        after = mib_violation(case, candidate)
                        if after >= before:
                            continue
                        key = tuple(
                            round(float(value), 10)
                            for value in candidate.reshape(-1).tolist()
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        found.append(
                            MibPatchCandidate(
                                placement=candidate,
                                group_index=group_index,
                                anchor_member=anchor,
                                target_member=target,
                                members=members_patch,
                                target_shape=shape,
                                mib_before=before,
                                mib_after=after,
                            )
                        )
                        if len(found) >= max_candidates:
                            return tuple(found)
    return tuple(found)


def _repack_member(
    source: Tensor,
    members: tuple[int, ...],
    target: int,
    shape: tuple[float, float],
    *,
    protected: Tensor,
    boundary_bits: Tensor,
    global_bounds: tuple[float, float, float, float],
) -> tuple[Tensor, ...]:
    left, bottom, right, top = _bounds(source[list(members)])
    width, height = shape
    if width > right - left + _EPS or height > top - bottom + _EPS:
        return ()
    target_center = source[target, :2] + 0.5 * source[target, 2:4]
    x_center = min(max(float(target_center[0]) - 0.5 * width, left), right - width)
    y_center = min(max(float(target_center[1]) - 0.5 * height, bottom), top - height)
    placements = (
        (x_center, y_center),
        (left, bottom),
        (right - width, bottom),
        (left, top - height),
        (right - width, top - height),
    )
    patch_bounds = (left, bottom, right, top)
    obstacles = [
        source[index]
        for index in members
        if index != target and bool(protected[index])
    ]
    centers = source[:, :2] + 0.5 * source[:, 2:4]
    areas = source[:, 2] * source[:, 3]
    output: list[Tensor] = []
    seen: set[tuple[float, ...]] = set()
    for x, y in placements:
        target_box = source.new_tensor((x, y, width, height))
        if _overlaps_any(target_box, obstacles):
            continue
        obstacle_rows = [*obstacles, target_box]
        obstacle_tensor = (
            torch.stack(obstacle_rows)
            if obstacle_rows
            else source.new_empty((0, 4))
        )
        free_regions = _free_rectangles(patch_bounds, obstacle_tensor, transpose=False)
        movable = [
            member
            for member in members
            if member != target and not bool(protected[member])
        ]
        items = [
            _Item(
                (member,),
                float(areas[member]),
                (float(centers[member, 0]), float(centers[member, 1])),
            )
            for member in movable
        ]
        allocation = _allocate_items(
            items,
            free_regions,
            boundary_bits=boundary_bits,
            outline=global_bounds,
        )
        if allocation is None:
            continue
        candidate = source.clone()
        candidate[target] = target_box
        rectangles: dict[int, tuple[float, float, float, float]] = {}
        for region, assigned in zip(free_regions, allocation, strict=True):
            if assigned:
                _place_items(
                    assigned,
                    region,
                    rectangles,
                    areas=areas,
                    centers=centers,
                    axis_mode="long",
                    whitespace_side="right",
                )
        if len(rectangles) != len(movable):
            continue
        for member, rect in rectangles.items():
            x0, y0, x1, y1 = rect
            candidate[member] = candidate.new_tensor((x0, y0, x1 - x0, y1 - y0))
        key = tuple(round(float(value), 10) for value in candidate.reshape(-1).tolist())
        if key not in seen:
            seen.add(key)
            output.append(candidate)
    return tuple(output)


def _anchor_shapes(
    boxes: Tensor,
    areas: Tensor,
    hard: Tensor,
    members: tuple[int, ...],
) -> tuple[tuple[int, tuple[float, float]], ...]:
    hard_members = [member for member in members if bool(hard[member])]
    if hard_members:
        anchor = hard_members[0]
        shape = tuple(float(value) for value in boxes[anchor, 2:4].tolist())
        if all(_same_shape(boxes[member, 2:4], shape) for member in hard_members):
            return ((anchor, shape),)
        return ()

    counts: dict[tuple[float, float], list[int]] = {}
    for member in members:
        key = tuple(round(float(value), 4) for value in boxes[member, 2:4].tolist())
        counts.setdefault(key, []).append(member)
    ranked = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0], item[1]))
    candidates = []
    for _, anchors in ranked[:2]:
        anchor = anchors[0]
        shape = tuple(float(value) for value in boxes[anchor, 2:4].tolist())
        if _area_compatible(areas[list(members)], shape):
            candidates.append((anchor, shape))
    return tuple(candidates)


def _same_shape(values: Tensor, shape: tuple[float, float]) -> bool:
    return all(abs(float(value) - expected) <= 5.0e-5 for value, expected in zip(values, shape))


def _shape_distance(values: Tensor, shape: tuple[float, float]) -> float:
    return abs(float(values[0]) - shape[0]) + abs(float(values[1]) - shape[1])


def _area_compatible(areas: Tensor, shape: tuple[float, float]) -> bool:
    target = shape[0] * shape[1]
    relative = torch.abs(areas - target) / torch.clamp(areas, min=1.0e-12)
    return bool((relative <= 1.0e-2).all())


def _resized_bounds(box: Tensor, shape: tuple[float, float]) -> tuple[float, float, float, float]:
    center = box[:2] + 0.5 * box[2:4]
    return (
        float(center[0] - 0.5 * shape[0]),
        float(center[1] - 0.5 * shape[1]),
        float(center[0] + 0.5 * shape[0]),
        float(center[1] + 0.5 * shape[1]),
    )


def _overlaps_any(box: Tensor, obstacles: list[Tensor]) -> bool:
    return any(
        min(float(box[0] + box[2]), float(other[0] + other[2]))
        - max(float(box[0]), float(other[0]))
        > _EPS
        and min(float(box[1] + box[3]), float(other[1] + other[3]))
        - max(float(box[1]), float(other[1]))
        > _EPS
        for other in obstacles
    )


def _membership(case: Any, n: int) -> Tensor:
    value = _field(case, "mib_membership")
    if value is None:
        return torch.empty((0, n), dtype=torch.bool)
    membership = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if membership.ndim != 2 or membership.shape[1] != n:
        raise ValueError("mib_membership must have shape [M,N]")
    return membership


def _mask(case: Any, name: str, n: int) -> Tensor:
    value = _field(case, name)
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(-1)
    if mask.numel() != n:
        raise ValueError(f"{name} must have shape [N]")
    return mask


def _areas(case: Any, boxes: Tensor) -> Tensor:
    value = _field(case, "area")
    if value is None:
        return boxes[:, 2] * boxes[:, 3]
    areas = torch.as_tensor(value, dtype=torch.float64, device="cpu").reshape(-1)
    if areas.numel() != boxes.shape[0]:
        raise ValueError("area must have shape [N]")
    return areas


def _boundary_bits(case: Any, n: int) -> Tensor:
    value = _field(case, "boundary_bits")
    if value is None:
        return torch.zeros((n, 4), dtype=torch.bool)
    bits = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if bits.shape != (n, 4):
        raise ValueError("boundary_bits must have shape [N,4]")
    return bits


def _one_hot(n: int, index: int) -> Tensor:
    result = torch.zeros(n, dtype=torch.bool)
    result[index] = True
    return result


def _field(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


__all__ = ["MibPatchCandidate", "mib_anchor_patch_candidates"]
