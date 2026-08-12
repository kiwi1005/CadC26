"""Exact-area recursive slicing candidates for latent FloorSet outlines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from hcfp.case import FloorplanCase
from hcfp.constraints.mib_shapes import resolve_mib_shapes


Tensor = torch.Tensor
_EPS = 1.0e-6
_MICRO_GUTTER = 2.0e-6


@dataclass(frozen=True)
class _Item:
    members: tuple[int, ...]
    area: float
    center: tuple[float, float]
    whitespace: bool = False


def exact_treemap_candidates(
    case: FloorplanCase,
    reference: Tensor,
    hypotheses: Sequence[Any],
    *,
    count: int,
    area_slack: float = 1.0,
) -> tuple[Tensor, tuple[dict[str, object], ...]]:
    """Use learned order plus exact slicing ratios to make dense seeds.

    ``area_slack`` is an opt-in contest experiment.  Values below one reduce
    only ordinary movable block areas before slicing; fixed, preplaced, and
    MIB members retain their exact target areas/shapes.  The default remains
    the historical exact-area behavior.
    """

    if count <= 0:
        return reference.new_empty((0, case.n, 4)), ()
    if not math.isfinite(area_slack) or not 0.0 < area_slack <= 1.0:
        raise ValueError("area_slack must be finite and in (0, 1]")
    reference = torch.as_tensor(reference, dtype=torch.float32, device=case.area.device)
    if reference.shape != (case.n, 4):
        raise ValueError("reference must have shape [N,4]")
    packing_areas = case.area.detach().clone()
    if area_slack != 1.0:
        hard = case.fixed_mask | case.preplaced_mask
        if case.mib_membership.numel():
            mib_members = case.mib_membership.any(dim=0)
        else:
            mib_members = torch.zeros_like(hard)
        ordinary = ~hard & ~mib_members
        packing_areas[ordinary] *= area_slack

    variants = (
        ("left", "long"),
        ("right", "long"),
        ("bottom", "long"),
        ("top", "long"),
        ("left", "x"),
        ("right", "y"),
        ("bottom", "x"),
        ("top", "y"),
    )
    candidates: list[Tensor] = []
    records: list[dict[str, object]] = []
    for hypothesis in hypotheses:
        try:
            bounds = tuple(float(value) for value in hypothesis.bounds)
            confidence = float(hypothesis.confidence)
        except (AttributeError, TypeError, ValueError):
            continue
        if len(bounds) != 4 or confidence < 0.5:
            continue
        left, bottom, right, top = bounds
        if not all(math.isfinite(value) for value in bounds):
            continue
        outline_area = (right - left) * (top - bottom)
        block_area = float(case.area.sum())
        if min(right - left, top - bottom) <= 0.0 or outline_area + _EPS < block_area:
            continue
        for whitespace_side, axis_mode in variants:
            candidate, whitespace_area, obstacle_aware, mib_groups = _pack_candidate(
                case,
                reference,
                bounds,
                whitespace_side=whitespace_side,
                axis_mode=axis_mode,
                packing_areas=packing_areas,
            )
            candidates.append(candidate)
            records.append(
                {
                    "hypothesis_id": str(
                        getattr(hypothesis, "hypothesis_id", "unknown")
                    ),
                    "outline_bounds": bounds,
                    "outline_confidence": confidence,
                    "whitespace_side": whitespace_side,
                    "whitespace_area": whitespace_area,
                    "axis_mode": axis_mode,
                    "area_slack": area_slack,
                    "obstacle_aware": obstacle_aware,
                    "mib_constructed_groups": mib_groups,
                }
            )
            if len(candidates) >= count:
                return torch.stack(candidates), tuple(records)
    if not candidates:
        return reference.new_empty((0, case.n, 4)), ()
    return torch.stack(candidates), tuple(records)


def _pack_candidate(
    case: FloorplanCase,
    reference: Tensor,
    bounds: tuple[float, float, float, float],
    *,
    whitespace_side: str,
    axis_mode: str,
    packing_areas: Tensor | None = None,
) -> tuple[Tensor, float, bool, tuple[int, ...]]:
    centers = reference[:, :2] + 0.5 * reference[:, 2:4]
    areas = (
        (case.area if packing_areas is None else packing_areas)
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    centers_cpu = centers.detach().to(device="cpu", dtype=torch.float64)

    left, bottom, right, top = bounds
    whitespace_area = max(0.0, (right - left) * (top - bottom) - float(areas.sum()))
    hard = (case.fixed_mask | case.preplaced_mask).detach().to(device="cpu")
    constrained = _place_constraint_obstacles(case, reference, bounds)
    if constrained is None:
        obstacles = _place_hard_obstacles(case, reference, bounds)
        occupied = hard
        mib_groups: tuple[int, ...] = ()
    else:
        obstacles, occupied, mib_groups = constrained

    if obstacles is not None and bool(occupied.any()):
        free = _free_rectangles(
            bounds,
            obstacles[occupied],
            transpose=axis_mode == "y",
        )
        movable = ~occupied
        items = _compound_items(case, movable, areas, centers_cpu)
        allocation = _allocate_items(
            items,
            free,
            boundary_bits=case.boundary_bits,
            outline=bounds,
        )
        if allocation is None:
            items = [
                _item(
                    (int(index),),
                    areas,
                    centers_cpu,
                )
                for index in torch.nonzero(movable).reshape(-1)
            ]
            allocation = _allocate_items(
                items,
                free,
                boundary_bits=case.boundary_bits,
                outline=bounds,
            )
        if allocation is not None:
            rectangles: dict[int, tuple[float, float, float, float]] = {}
            for region, region_items in zip(free, allocation, strict=True):
                if region_items:
                    _place_items(
                        region_items,
                        region,
                        rectangles,
                        areas=areas,
                        centers=centers_cpu,
                        axis_mode=axis_mode,
                        whitespace_side=whitespace_side,
                    )
            if len(rectangles) == int(movable.sum()):
                candidate = reference.new_empty((case.n, 4))
                for index in torch.nonzero(movable).reshape(-1).tolist():
                    candidate[index] = _xywh_tensor(
                        candidate,
                        rectangles[index],
                        boundary_bits=case.boundary_bits[index],
                        outline=bounds,
                    )
                candidate[occupied.to(device=candidate.device)] = obstacles[
                    occupied
                ].to(device=candidate.device, dtype=candidate.dtype)
                return candidate, whitespace_area, True, mib_groups

    items = _compound_items(
        case,
        torch.ones(case.n, dtype=torch.bool),
        areas,
        centers_cpu,
    )
    if whitespace_area > _EPS:
        items.append(
            _Item(
                members=(),
                area=whitespace_area,
                center=_whitespace_center(bounds, whitespace_side),
                whitespace=True,
            )
        )

    unit_rectangles: dict[int, tuple[float, float, float, float]] = {}
    _partition(
        items,
        bounds,
        unit_rectangles,
        axis_mode=axis_mode,
        whitespace_side=whitespace_side,
        depth=0,
    )
    rectangles: dict[int, tuple[float, float, float, float]] = {}
    for item_index, item in enumerate(items):
        if item.whitespace:
            continue
        rectangle = unit_rectangles[item_index]
        if len(item.members) == 1:
            rectangles[item.members[0]] = rectangle
            continue
        member_items = [
            _item(
                (index,),
                areas,
                centers_cpu,
            )
            for index in item.members
        ]
        member_rectangles: dict[int, tuple[float, float, float, float]] = {}
        _partition(
            member_items,
            rectangle,
            member_rectangles,
            axis_mode=axis_mode,
            whitespace_side=whitespace_side,
            depth=1,
        )
        for member_offset, member in enumerate(member_items):
            rectangles[member.members[0]] = member_rectangles[member_offset]

    candidate = reference.new_empty((case.n, 4))
    for index in range(case.n):
        candidate[index] = _xywh_tensor(
            candidate,
            rectangles[index],
            boundary_bits=case.boundary_bits[index],
            outline=bounds,
        )

    hard_shape = (case.fixed_mask | case.preplaced_mask).to(device=candidate.device)
    if bool(hard_shape.any()):
        target_wh = case.target[:, 2:4].to(
            device=candidate.device, dtype=candidate.dtype
        )
        allocated_center = candidate[:, :2] + 0.5 * candidate[:, 2:4]
        candidate[hard_shape, 2:4] = target_wh[hard_shape]
        candidate[hard_shape, :2] = (
            allocated_center[hard_shape] - 0.5 * target_wh[hard_shape]
        )
        max_x = (right - candidate[hard_shape, 2]).clamp_min(left)
        max_y = (top - candidate[hard_shape, 3]).clamp_min(bottom)
        candidate[hard_shape, 0] = torch.maximum(
            candidate[hard_shape, 0].clamp_min(left),
            max_x.minimum(candidate[hard_shape, 0]),
        )
        candidate[hard_shape, 1] = torch.maximum(
            candidate[hard_shape, 1].clamp_min(bottom),
            max_y.minimum(candidate[hard_shape, 1]),
        )
    preplaced = case.preplaced_mask.to(device=candidate.device)
    if bool(preplaced.any()):
        candidate[preplaced] = case.target.to(
            device=candidate.device, dtype=candidate.dtype
        )[preplaced]
    return candidate, whitespace_area, False, ()


def _place_constraint_obstacles(
    case: FloorplanCase,
    reference: Tensor,
    bounds: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    """Place compatible MIB rows/columns before exact slicing."""

    hard = (case.fixed_mask | case.preplaced_mask).detach().to(device="cpu")
    preplaced = case.preplaced_mask.detach().to(device="cpu")
    result = reference.detach().to(device="cpu", dtype=torch.float32).clone()
    target = case.target.detach().to(device="cpu", dtype=torch.float32)
    hard_wh = result[:, 2:4].clone()
    hard_wh[hard] = target[hard, 2:4]
    resolution = resolve_mib_shapes(
        case.area,
        case.mib_membership,
        proposed_wh=result[:, 2:4],
        hard_mask=hard,
        hard_wh=hard_wh,
    )
    result[preplaced] = target[preplaced]
    occupied = preplaced.clone()
    placed = [result[index].clone() for index in torch.nonzero(preplaced).reshape(-1)]
    if _boxes_overlap(placed):
        return None

    units: list[tuple[tuple[int, ...], int | None]] = []
    mib_members: set[int] = set()
    for group in resolution.groups:
        if not group.compatible or any(
            bool(preplaced[index]) for index in group.members
        ):
            continue
        units.append((group.members, group.group))
        mib_members.update(group.members)
        shape = resolution.shapes[group.members[0]]
        result[list(group.members), 2:4] = shape
    for index in torch.nonzero(hard & ~preplaced).reshape(-1).tolist():
        if index not in mib_members:
            result[index, 2:4] = target[index, 2:4]
            units.append(((index,), None))

    centers = reference[:, :2] + 0.5 * reference[:, 2:4]
    units.sort(
        key=lambda unit: (
            -sum(float(result[index, 2] * result[index, 3]) for index in unit[0]),
            unit[0],
        )
    )
    constructed: list[int] = []
    for members, group_index in units:
        choice = _place_unit(case, result, centers, members, placed, bounds)
        if choice is None:
            return None
        for index, box in choice.items():
            result[index] = box
            occupied[index] = True
            placed.append(box.clone())
        if group_index is not None:
            constructed.append(group_index)
    return result, occupied, tuple(constructed)


def _place_unit(
    case: FloorplanCase,
    boxes: Tensor,
    reference_centers: Tensor,
    members: tuple[int, ...],
    placed: list[Tensor],
    bounds: tuple[float, float, float, float],
) -> dict[int, Tensor] | None:
    width = float(boxes[members[0], 2])
    height = float(boxes[members[0], 3])
    if len(members) == 1:
        layouts = ((members, "row"),)
    else:
        x_order = tuple(
            sorted(
                members, key=lambda index: (float(reference_centers[index, 0]), index)
            )
        )
        y_order = tuple(
            sorted(
                members, key=lambda index: (float(reference_centers[index, 1]), index)
            )
        )
        layouts = tuple(
            dict.fromkeys(
                (
                    (x_order, "row"),
                    (x_order[::-1], "row"),
                    (y_order, "column"),
                    (y_order[::-1], "column"),
                )
            )
        )

    left, bottom, right, top = bounds
    best: tuple[tuple[float, float, float], dict[int, Tensor]] | None = None
    for order, orientation in layouts:
        footprint_w = (
            width * len(order) + _MICRO_GUTTER * (len(order) - 1)
            if orientation == "row"
            else width
        )
        footprint_h = (
            height
            if orientation == "row"
            else height * len(order) + _MICRO_GUTTER * (len(order) - 1)
        )
        x_values = {left, right - footprint_w}
        y_values = {bottom, top - footprint_h}
        for obstacle in placed:
            x_values.update(
                (
                    float(obstacle[0] - footprint_w - _MICRO_GUTTER),
                    float(obstacle[0] + obstacle[2] + _MICRO_GUTTER),
                )
            )
            y_values.update(
                (
                    float(obstacle[1] - footprint_h - _MICRO_GUTTER),
                    float(obstacle[1] + obstacle[3] + _MICRO_GUTTER),
                )
            )
        for x in x_values:
            for y in y_values:
                if (
                    x < left - _EPS
                    or y < bottom - _EPS
                    or x + footprint_w > right + _EPS
                    or y + footprint_h > top + _EPS
                ):
                    continue
                proposed: dict[int, Tensor] = {}
                for offset, index in enumerate(order):
                    px = x + (
                        offset * (width + _MICRO_GUTTER)
                        if orientation == "row"
                        else 0.0
                    )
                    py = y + (
                        offset * (height + _MICRO_GUTTER)
                        if orientation == "column"
                        else 0.0
                    )
                    proposed[index] = boxes.new_tensor((px, py, width, height))
                if _proposed_overlaps(proposed.values(), placed):
                    continue
                boundary_miss = 0.0
                distance = 0.0
                for index, box in proposed.items():
                    bits = case.boundary_bits[index].detach().to(device="cpu")
                    boundary_miss += (
                        float(bits[0]) * abs(float(box[0]) - left)
                        + float(bits[1]) * abs(float(box[0] + box[2]) - right)
                        + float(bits[2]) * abs(float(box[1] + box[3]) - top)
                        + float(bits[3]) * abs(float(box[1]) - bottom)
                    )
                    distance += abs(
                        float(box[0] + 0.5 * box[2])
                        - float(reference_centers[index, 0])
                    )
                    distance += abs(
                        float(box[1] + 0.5 * box[3])
                        - float(reference_centers[index, 1])
                    )
                score = (boundary_miss, distance, x + y)
                if best is None or score < best[0]:
                    best = (score, proposed)
    return None if best is None else best[1]


def _place_hard_obstacles(
    case: FloorplanCase,
    reference: Tensor,
    bounds: tuple[float, float, float, float],
) -> Tensor | None:
    hard = (case.fixed_mask | case.preplaced_mask).detach().to(device="cpu")
    preplaced = case.preplaced_mask.detach().to(device="cpu")
    fixed_movable = (case.fixed_mask & ~case.preplaced_mask).detach().to(device="cpu")
    result = reference.detach().to(device="cpu", dtype=torch.float32).clone()
    result[hard, 2:4] = case.target.detach().to(device="cpu")[hard, 2:4]
    result[preplaced] = case.target.detach().to(device="cpu")[preplaced]
    placed = [result[index].clone() for index in torch.nonzero(preplaced).reshape(-1)]
    if _boxes_overlap(placed):
        return None

    order = sorted(
        torch.nonzero(fixed_movable, as_tuple=False).reshape(-1).tolist(),
        key=lambda index: (
            -float(result[index, 2] * result[index, 3]),
            index,
        ),
    )
    for index in order:
        width, height = float(result[index, 2]), float(result[index, 3])
        reference_center = (
            float(reference[index, 0] + 0.5 * reference[index, 2]),
            float(reference[index, 1] + 0.5 * reference[index, 3]),
        )
        left, bottom, right, top = bounds
        x_values = {left, right - width}
        y_values = {bottom, top - height}
        for obstacle in placed:
            x_values.update(
                (
                    float(obstacle[0] - width - _MICRO_GUTTER),
                    float(obstacle[0] + obstacle[2] + _MICRO_GUTTER),
                )
            )
            y_values.update(
                (
                    float(obstacle[1] - height - _MICRO_GUTTER),
                    float(obstacle[1] + obstacle[3] + _MICRO_GUTTER),
                )
            )
        candidates = []
        for x in x_values:
            for y in y_values:
                box = result.new_tensor((x, y, width, height))
                if (
                    x >= left - _EPS
                    and y >= bottom - _EPS
                    and x + width <= right + _EPS
                    and y + height <= top + _EPS
                    and not _boxes_overlap([*placed, box])
                ):
                    candidates.append((x, y))
        if not candidates:
            return None
        boundary = case.boundary_bits[index].detach().to(device="cpu")

        def score(position: tuple[float, float]) -> tuple[float, float, float]:
            x, y = position
            boundary_miss = (
                float(boundary[0]) * abs(x - left)
                + float(boundary[1]) * abs(x + width - right)
                + float(boundary[2]) * abs(y + height - top)
                + float(boundary[3]) * abs(y - bottom)
            )
            distance = abs(x + 0.5 * width - reference_center[0]) + abs(
                y + 0.5 * height - reference_center[1]
            )
            return (boundary_miss, distance, x + y)

        x, y = min(candidates, key=score)
        result[index] = result.new_tensor((x, y, width, height))
        placed.append(result[index].clone())
    return result


def _compound_items(
    case: FloorplanCase,
    eligible: Tensor,
    areas: Tensor,
    centers: Tensor,
) -> list[_Item]:
    items: list[_Item] = []
    assigned: set[int] = set()
    for membership in case.group_membership.detach().to(device="cpu", dtype=torch.bool):
        members = tuple(
            int(index)
            for index in torch.nonzero(membership & eligible, as_tuple=False).reshape(
                -1
            )
            if int(index) not in assigned
        )
        if len(members) < 2:
            continue
        assigned.update(members)
        items.append(
            _item(
                members,
                areas,
                centers,
            )
        )
    for index in torch.nonzero(eligible, as_tuple=False).reshape(-1).tolist():
        if index not in assigned:
            items.append(
                _item(
                    (index,),
                    areas,
                    centers,
                )
            )
    return items


def _place_items(
    items: list[_Item],
    bounds: tuple[float, float, float, float],
    rectangles: dict[int, tuple[float, float, float, float]],
    *,
    areas: Tensor,
    centers: Tensor,
    axis_mode: str,
    whitespace_side: str,
) -> None:
    capacity = _rectangle_area(bounds)
    used = sum(item.area for item in items)
    packed = list(items)
    if capacity > used + _EPS:
        packed.append(
            _Item(
                (), capacity - used, _whitespace_center(bounds, whitespace_side), True
            )
        )
    unit_rectangles: dict[int, tuple[float, float, float, float]] = {}
    _partition(
        packed,
        bounds,
        unit_rectangles,
        axis_mode=axis_mode,
        whitespace_side=whitespace_side,
        depth=0,
    )
    for item_index, item in enumerate(packed):
        if item.whitespace:
            continue
        rectangle = unit_rectangles[item_index]
        if len(item.members) == 1:
            rectangles[item.members[0]] = rectangle
            continue
        member_items = [
            _item(
                (member,),
                areas,
                centers,
            )
            for member in item.members
        ]
        member_rectangles: dict[int, tuple[float, float, float, float]] = {}
        _partition(
            member_items,
            rectangle,
            member_rectangles,
            axis_mode=axis_mode,
            whitespace_side=whitespace_side,
            depth=1,
        )
        for offset, member in enumerate(item.members):
            rectangles[member] = member_rectangles[offset]


def _allocate_items(
    items: list[_Item],
    regions: list[tuple[float, float, float, float]],
    *,
    boundary_bits: Tensor | None = None,
    outline: tuple[float, float, float, float] | None = None,
) -> list[list[_Item]] | None:
    remaining = [_rectangle_area(region) for region in regions]
    allocation: list[list[_Item]] = [[] for _ in regions]
    for item in sorted(items, key=lambda value: (-value.area, value.members)):
        eligible = [
            index
            for index, capacity in enumerate(remaining)
            if capacity + _EPS >= item.area
        ]
        if not eligible:
            return None
        best = min(
            eligible,
            key=lambda index: (
                _boundary_region_misses(
                    item,
                    regions[index],
                    boundary_bits=boundary_bits,
                    outline=outline,
                ),
                _center_distance(item.center, regions[index]),
                remaining[index] - item.area,
                index,
            ),
        )
        allocation[best].append(item)
        remaining[best] -= item.area
    return allocation


def _boundary_region_misses(
    item: _Item,
    region: tuple[float, float, float, float],
    *,
    boundary_bits: Tensor | None,
    outline: tuple[float, float, float, float] | None,
) -> int:
    if boundary_bits is None or outline is None:
        return 0
    members = torch.tensor(item.members, dtype=torch.long)
    bits = torch.as_tensor(boundary_bits, dtype=torch.bool, device="cpu")[members].any(
        dim=0
    )
    return sum(
        (
            int(bool(bits[0]) and abs(region[0] - outline[0]) > _EPS),
            int(bool(bits[1]) and abs(region[2] - outline[2]) > _EPS),
            int(bool(bits[2]) and abs(region[3] - outline[3]) > _EPS),
            int(bool(bits[3]) and abs(region[1] - outline[1]) > _EPS),
        )
    )


def _free_rectangles(
    bounds: tuple[float, float, float, float],
    obstacles: Tensor,
    *,
    transpose: bool,
) -> list[tuple[float, float, float, float]]:
    if transpose:
        swapped = obstacles[:, [1, 0, 3, 2]]
        transposed = _free_rectangles(
            (bounds[1], bounds[0], bounds[3], bounds[2]),
            swapped,
            transpose=False,
        )
        return [(bottom, left, top, right) for left, bottom, right, top in transposed]

    left, bottom, right, top = bounds
    boxes = [tuple(float(value) for value in row) for row in obstacles]
    for index, (x, y, width, height) in enumerate(boxes):
        if (
            x < left - _EPS
            or y < bottom - _EPS
            or x + width > right + _EPS
            or y + height > top + _EPS
        ):
            return []
        for other in boxes[index + 1 :]:
            ox, oy, ow, oh = other
            if (
                min(x + width, ox + ow) - max(x, ox) > _EPS
                and min(y + height, oy + oh) - max(y, oy) > _EPS
            ):
                return []

    xs = sorted(
        {left, right, *(value for box in boxes for value in (box[0], box[0] + box[2]))}
    )
    active: dict[tuple[float, float], tuple[float, float, float, float]] = {}
    result: list[tuple[float, float, float, float]] = []
    for x0, x1 in zip(xs, xs[1:], strict=False):
        blocked = sorted(
            (y, y + height)
            for x, y, width, height in boxes
            if x < x1 - _EPS and x + width > x0 + _EPS
        )
        free_intervals: list[tuple[float, float]] = []
        cursor = bottom
        for y0, y1 in blocked:
            if y0 > cursor + _EPS:
                free_intervals.append((cursor, y0))
            cursor = max(cursor, y1)
        if cursor < top - _EPS:
            free_intervals.append((cursor, top))
        current = set(free_intervals)
        for interval, rectangle in tuple(active.items()):
            if interval not in current:
                result.append(rectangle)
                del active[interval]
        for y0, y1 in free_intervals:
            previous = active.get((y0, y1))
            active[(y0, y1)] = (
                (previous[0], y0, x1, y1) if previous else (x0, y0, x1, y1)
            )
    result.extend(active.values())
    return [rectangle for rectangle in result if _rectangle_area(rectangle) > _EPS]


def _rectangle_area(bounds: tuple[float, float, float, float]) -> float:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _boxes_overlap(boxes: list[Tensor]) -> bool:
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            overlap_x = min(
                float(first[0] + first[2]), float(second[0] + second[2])
            ) - max(float(first[0]), float(second[0]))
            overlap_y = min(
                float(first[1] + first[3]), float(second[1] + second[3])
            ) - max(float(first[1]), float(second[1]))
            if overlap_x > _EPS and overlap_y > _EPS:
                return True
    return False


def _proposed_overlaps(proposed: Any, placed: list[Tensor]) -> bool:
    boxes = list(proposed)
    if _boxes_overlap(boxes):
        return True
    return any(_pair_overlaps(first, second) for first in boxes for second in placed)


def _pair_overlaps(first: Tensor, second: Tensor) -> bool:
    overlap_x = min(float(first[0] + first[2]), float(second[0] + second[2])) - max(
        float(first[0]), float(second[0])
    )
    overlap_y = min(float(first[1] + first[3]), float(second[1] + second[3])) - max(
        float(first[1]), float(second[1])
    )
    return overlap_x > _EPS and overlap_y > _EPS


def _center_distance(
    center: tuple[float, float], bounds: tuple[float, float, float, float]
) -> float:
    return abs(center[0] - 0.5 * (bounds[0] + bounds[2])) + abs(
        center[1] - 0.5 * (bounds[1] + bounds[3])
    )


def _xywh_tensor(
    template: Tensor,
    bounds: tuple[float, float, float, float],
    *,
    boundary_bits: Tensor | None = None,
    outline: tuple[float, float, float, float] | None = None,
) -> Tensor:
    edges = template.new_tensor(bounds)
    lower = edges.new_full((2,), _MICRO_GUTTER)
    upper = edges.new_full((2,), _MICRO_GUTTER)
    if boundary_bits is not None and outline is not None:
        bits = torch.as_tensor(boundary_bits, dtype=torch.bool, device="cpu")
        if bool(bits[0]) and abs(bounds[0] - outline[0]) <= _EPS:
            lower[0] = 0.0
        if bool(bits[1]) and abs(bounds[2] - outline[2]) <= _EPS:
            upper[0] = 0.0
        if bool(bits[2]) and abs(bounds[3] - outline[3]) <= _EPS:
            upper[1] = 0.0
        if bool(bits[3]) and abs(bounds[1] - outline[1]) <= _EPS:
            lower[1] = 0.0
    origin = edges[0:2] + lower
    dimensions = (edges[2:4] - edges[0:2] - lower - upper).clamp_min(_EPS)
    dimensions = torch.nextafter(dimensions, torch.zeros_like(dimensions))
    return torch.cat((origin, dimensions))


def _item(
    members: tuple[int, ...],
    areas: Tensor,
    centers: Tensor,
) -> _Item:
    index = torch.tensor(members, dtype=torch.long)
    member_area = areas[index]
    total = float(member_area.sum())
    center = (centers[index] * member_area[:, None]).sum(dim=0) / total
    return _Item(members, total, (float(center[0]), float(center[1])))


def _partition(
    items: list[_Item],
    bounds: tuple[float, float, float, float],
    output: dict[int, tuple[float, float, float, float]],
    *,
    axis_mode: str,
    whitespace_side: str,
    depth: int,
) -> None:
    indexed = list(enumerate(items))

    def visit(
        subset: list[tuple[int, _Item]],
        rectangle: tuple[float, float, float, float],
        level: int,
    ) -> None:
        if len(subset) == 1:
            output[subset[0][0]] = rectangle
            return
        left, bottom, right, top = rectangle
        axis = _axis(rectangle, axis_mode, whitespace_side, level)
        subset.sort(key=lambda entry: (entry[1].center[axis], entry[0]))
        total = sum(item.area for _, item in subset)
        cumulative = 0.0
        best_index = 1
        best_error = math.inf
        for split in range(1, len(subset)):
            cumulative += subset[split - 1][1].area
            error = abs(cumulative - 0.5 * total)
            if error < best_error:
                best_index, best_error = split, error
        first, second = subset[:best_index], subset[best_index:]
        first_area = sum(item.area for _, item in first)
        ratio = min(1.0 - _EPS, max(_EPS, first_area / total))
        if axis == 0:
            cut = left + (right - left) * ratio
            first_rect = (left, bottom, cut, top)
            second_rect = (cut, bottom, right, top)
        else:
            cut = bottom + (top - bottom) * ratio
            first_rect = (left, bottom, right, cut)
            second_rect = (left, cut, right, top)
        visit(first, first_rect, level + 1)
        visit(second, second_rect, level + 1)

    visit(indexed, bounds, depth)


def _axis(
    bounds: tuple[float, float, float, float],
    mode: str,
    whitespace_side: str,
    depth: int,
) -> int:
    if depth == 0:
        return 0 if whitespace_side in {"left", "right"} else 1
    if mode == "x":
        return depth % 2
    if mode == "y":
        return (depth + 1) % 2
    left, bottom, right, top = bounds
    return 0 if right - left >= top - bottom else 1


def _whitespace_center(
    bounds: tuple[float, float, float, float], side: str
) -> tuple[float, float]:
    left, bottom, right, top = bounds
    midpoint = (0.5 * (left + right), 0.5 * (bottom + top))
    if side == "left":
        return (-math.inf, midpoint[1])
    if side == "right":
        return (math.inf, midpoint[1])
    if side == "bottom":
        return (midpoint[0], -math.inf)
    return (midpoint[0], math.inf)


__all__ = ["exact_treemap_candidates"]
