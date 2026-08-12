"""Fixed-frame boundary witness challengers for dense placements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.contact_patch import _bounds, _closed_patch, _protected, _xywh
from hcfp.treemap import _Item, _allocate_items, _free_rectangles, _place_items
from hcfp.verify import boundary_missing, verify_feasible


Tensor = torch.Tensor
_EPS = 1.0e-9
_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class BoundarySkeletonCandidate:
    placement: Tensor
    block: int
    required_sides: tuple[str, ...]
    members: tuple[int, ...]
    missing_before: int
    missing_after: int


def boundary_skeleton_candidates(
    case: Any,
    placements: Any,
    *,
    verify_case: Any | None = None,
    patch_sizes: tuple[int, ...] = (4, 8, 12, 16),
    max_candidates: int = 16,
) -> tuple[BoundarySkeletonCandidate, ...]:
    """Re-slice a closed edge patch with a required block as its side witness."""

    boxes = torch.as_tensor(placements, dtype=torch.float64, device="cpu").clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("placements must have shape [N,4]")
    if max_candidates <= 0:
        return ()
    bits = torch.as_tensor(case.boundary_bits, dtype=torch.bool, device="cpu")
    if bits.shape != (len(boxes), 4):
        raise ValueError("boundary_bits must have shape [N,4]")
    missing = boundary_missing(case, boxes)
    missing_before = int(torch.count_nonzero(missing))
    if missing_before == 0:
        return ()
    protected = _protected(case, len(boxes))
    global_bounds = _bounds(boxes)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    sizes = tuple(sorted({int(size) for size in patch_sizes if int(size) >= 2}))
    found: list[BoundarySkeletonCandidate] = []
    seen: set[tuple[float, ...]] = set()

    targets = torch.nonzero(missing != 0, as_tuple=False).reshape(-1).tolist()
    targets.sort(key=lambda block: (-int(bits[block].sum()), block))
    for block in targets:
        if bool(protected[block]):
            continue
        required_sides = tuple(
            side for index, side in enumerate(_SIDES) if bool(bits[block, index])
        )
        witnesses = _witnesses(
            boxes,
            global_bounds,
            required_sides,
            centers[block],
            protected=protected,
        )
        if witnesses is None:
            continue
        distance = torch.abs(centers - centers[block]).sum(1)
        nearest = torch.argsort(distance, stable=True).tolist()
        for size in sizes:
            schemes: list[
                tuple[
                    tuple[int, ...],
                    tuple[tuple[float, float, float, float], ...],
                    tuple[float, float, float, float],
                ]
            ] = []
            seed = {
                block,
                *witnesses,
                *nearest[: max(0, size - 1 - len(witnesses))],
            }
            members = _closed_patch(boxes, seed, limit=size)
            if members is not None:
                patch_bounds = _bounds(boxes[list(members)])
                schemes.append((members, (patch_bounds,), patch_bounds))
            two_patch = _two_patch_scheme(
                boxes, block, witnesses, size, centers, protected
            )
            if two_patch is not None:
                schemes.append(two_patch)
            for members, patch_bounds, destination in schemes:
                if not _touches_required_edges(
                    destination, global_bounds, required_sides
                ):
                    continue
                candidate = _repack_with_witness(
                    boxes,
                    members,
                    block,
                    patch_bounds,
                    destination,
                    required_sides,
                    protected=protected,
                    boundary_bits=bits,
                    global_bounds=global_bounds,
                )
                if candidate is None or not verify_feasible(
                    verify_case or case, candidate
                ):
                    continue
                missing_after = int(
                    torch.count_nonzero(boundary_missing(case, candidate))
                )
                if missing_after >= missing_before:
                    continue
                key = tuple(round(float(value), 10) for value in candidate.reshape(-1))
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    BoundarySkeletonCandidate(
                        candidate,
                        block,
                        required_sides,
                        members,
                        missing_before,
                        missing_after,
                    )
                )
                if len(found) >= max_candidates:
                    return tuple(found)
    return tuple(found)


def _witnesses(
    boxes: Tensor,
    bounds: tuple[float, float, float, float],
    sides: tuple[str, ...],
    target_center: Tensor,
    *,
    protected: Tensor | None = None,
) -> tuple[int, ...] | None:
    left, bottom, right, top = bounds
    values = {
        "left": boxes[:, 0],
        "right": boxes[:, 0] + boxes[:, 2],
        "top": boxes[:, 1] + boxes[:, 3],
        "bottom": boxes[:, 1],
    }
    expected = {"left": left, "right": right, "top": top, "bottom": bottom}
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    selected: set[int] = set()
    for side in sides:
        candidates = torch.nonzero(
            torch.abs(values[side] - expected[side]) <= 1.0e-7,
            as_tuple=False,
        ).reshape(-1)
        if not candidates.numel():
            return None
        if protected is not None:
            movable = candidates[~protected[candidates]]
            if movable.numel():
                candidates = movable
        distance = torch.abs(centers[candidates] - target_center).sum(1)
        selected.add(int(candidates[int(torch.argmin(distance))]))
    return tuple(sorted(selected))


def _repack_with_witness(
    source: Tensor,
    members: tuple[int, ...],
    target: int,
    patch_bounds: tuple[tuple[float, float, float, float], ...],
    destination: tuple[float, float, float, float],
    required_sides: tuple[str, ...],
    *,
    protected: Tensor,
    boundary_bits: Tensor,
    global_bounds: tuple[float, float, float, float],
) -> Tensor | None:
    areas = source[:, 2] * source[:, 3]
    target_area = float(areas[target])
    protected_members = tuple(member for member in members if bool(protected[member]))
    obstacles = (
        source[list(protected_members)]
        if protected_members
        else source.new_empty((0, 4))
    )
    regions = _free_regions(patch_bounds, obstacles)
    eligible = [
        region
        for region in regions
        if _contained_in(region, destination)
        and _touches_required_edges(region, global_bounds, required_sides)
        and (region[2] - region[0]) * (region[3] - region[1]) >= target_area - _EPS
    ]
    if not eligible:
        return None
    center = source[target, :2] + 0.5 * source[target, 2:4]
    eligible.sort(
        key=lambda region: (
            abs(float(center[0]) - 0.5 * (region[0] + region[2]))
            + abs(float(center[1]) - 0.5 * (region[1] + region[3])),
            (region[2] - region[0]) * (region[3] - region[1]),
            region,
        ),
    )
    preferred_vertical = bool({"left", "right"} & set(required_sides))
    if {"left", "right"}.issubset(required_sides):
        preferred_vertical = False
    if {"top", "bottom"}.issubset(required_sides):
        preferred_vertical = True

    centers = source[:, :2] + 0.5 * source[:, 2:4]
    items = [
        _Item((member,), float(areas[member]), tuple(float(v) for v in centers[member]))
        for member in members
        if member != target and not bool(protected[member])
    ]
    for witness_region in eligible:
        for vertical in (preferred_vertical, not preferred_vertical):
            target_rect = _carve_witness(
                witness_region,
                target_area,
                required_sides,
                vertical=vertical,
            )
            if target_rect is None:
                continue
            all_obstacles = torch.cat(
                (obstacles, _xywh(source, target_rect).reshape(1, 4)), dim=0
            )
            free_regions = _free_regions(patch_bounds, all_obstacles)
            allocation = _allocate_items(
                items,
                free_regions,
                boundary_bits=boundary_bits,
                outline=global_bounds,
            )
            if allocation is None:
                continue
            candidate = source.clone()
            candidate[target] = _xywh(candidate, target_rect)
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
            for member, rectangle in rectangles.items():
                candidate[member] = _xywh(candidate, rectangle)
            return candidate
    return None


def _carve_witness(
    region: tuple[float, float, float, float],
    area: float,
    required_sides: tuple[str, ...],
    *,
    vertical: bool,
) -> tuple[float, float, float, float] | None:
    left, bottom, right, top = region
    width, height = right - left, top - bottom
    if vertical:
        target_width = area / height
        if target_width > width + _EPS:
            return None
        return (
            (right - target_width, bottom, right, top)
            if "right" in required_sides
            else (left, bottom, left + target_width, top)
        )
    target_height = area / width
    if target_height > height + _EPS:
        return None
    return (
        (left, top - target_height, right, top)
        if "top" in required_sides
        else (left, bottom, right, bottom + target_height)
    )


def _two_patch_scheme(
    boxes: Tensor,
    target: int,
    witnesses: tuple[int, ...],
    limit: int,
    centers: Tensor,
    protected: Tensor,
) -> (
    tuple[
        tuple[int, ...],
        tuple[tuple[float, float, float, float], ...],
        tuple[float, float, float, float],
    ]
    | None
):
    if limit < 4:
        return None
    source_members = (target,)
    destination_limit = limit - 1
    if destination_limit < len(witnesses):
        return None
    witness_center = centers[list(witnesses)].mean(0)
    destination_order = torch.argsort(
        torch.abs(centers - witness_center).sum(1), stable=True
    ).tolist()
    source_bounds = _bounds(boxes[list(source_members)])
    if all(not bool(protected[witness]) for witness in witnesses):
        destination_members = _closed_patch(
            boxes,
            set(witnesses),
            limit=destination_limit,
        )
        if destination_members is not None and target not in destination_members:
            destination_bounds = _bounds(boxes[list(destination_members)])
            if not _bounds_overlap(source_bounds, destination_bounds):
                members = tuple(sorted((*source_members, *destination_members)))
                return members, (source_bounds, destination_bounds), destination_bounds
    for neighbor in destination_order:
        if neighbor == target or neighbor in witnesses or bool(protected[neighbor]):
            continue
        destination_members = _closed_patch(
            boxes,
            {*witnesses, neighbor},
            limit=destination_limit,
        )
        if destination_members is None or target in destination_members:
            continue
        destination_bounds = _bounds(boxes[list(destination_members)])
        if _bounds_overlap(source_bounds, destination_bounds):
            continue
        members = tuple(sorted((*source_members, *destination_members)))
        return members, (source_bounds, destination_bounds), destination_bounds
    return None


def _free_regions(
    bounds: tuple[tuple[float, float, float, float], ...], obstacles: Tensor
) -> list[tuple[float, float, float, float]]:
    regions: list[tuple[float, float, float, float]] = []
    for patch in bounds:
        contained = []
        for obstacle in obstacles:
            left, bottom, right, top = patch
            x, y, width, height = (float(value) for value in obstacle)
            if (
                x >= left - _EPS
                and y >= bottom - _EPS
                and x + width <= right + _EPS
                and y + height <= top + _EPS
            ):
                contained.append(obstacle)
        patch_obstacles = (
            torch.stack(contained) if contained else obstacles.new_empty((0, 4))
        )
        regions.extend(_free_rectangles(patch, patch_obstacles, transpose=False))
    return regions


def _bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        min(first[2], second[2]) - max(first[0], second[0]) > _EPS
        and min(first[3], second[3]) - max(first[1], second[1]) > _EPS
    )


def _contained_in(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    return (
        inner[0] >= outer[0] - _EPS
        and inner[1] >= outer[1] - _EPS
        and inner[2] <= outer[2] + _EPS
        and inner[3] <= outer[3] + _EPS
    )


def _touches_required_edges(
    patch: tuple[float, float, float, float],
    global_bounds: tuple[float, float, float, float],
    sides: tuple[str, ...],
) -> bool:
    side_index = {"left": 0, "bottom": 1, "right": 2, "top": 3}
    return all(
        abs(patch[side_index[side]] - global_bounds[side_index[side]]) <= 1.0e-7
        for side in sides
    )


__all__ = ["BoundarySkeletonCandidate", "boundary_skeleton_candidates"]
