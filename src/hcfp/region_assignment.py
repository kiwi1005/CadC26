"""Obstacle-aware island-to-region assignment with one bounded topology split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from hcfp.island_relocation import detect_islands
from hcfp.treemap import (
    _Item,
    _allocate_items,
    _free_rectangles,
    _item,
    _place_constraint_obstacles,
    _place_items,
    _xywh_tensor,
)
from hcfp.verify import verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class RegionAssignmentRecord:
    hypothesis_id: str
    axis_mode: str
    split_members: tuple[int, ...]
    split_axis: str | None
    split_cut: int | None
    region_count: int


def obstacle_region_candidates(
    case: Any,
    reference: Tensor,
    hypotheses: Sequence[Any],
    *,
    count: int = 8,
) -> tuple[Tensor, tuple[RegionAssignmentRecord, ...]]:
    """Assign spatial islands to latent-outline free regions, splitting once if needed."""

    if count <= 0:
        return reference.new_empty((0, int(case.n), 4)), ()
    source = torch.as_tensor(reference, dtype=torch.float32, device=case.area.device)
    if source.shape != (int(case.n), 4):
        raise ValueError("reference must have shape [N,4]")
    candidates: list[Tensor] = []
    records: list[RegionAssignmentRecord] = []
    seen: set[tuple[float, ...]] = set()
    for hypothesis in hypotheses:
        try:
            bounds = tuple(float(value) for value in hypothesis.bounds)
            confidence = float(hypothesis.confidence)
            hypothesis_id = str(hypothesis.hypothesis_id)
        except (AttributeError, TypeError, ValueError):
            continue
        if len(bounds) != 4 or confidence < 0.5:
            continue
        for axis_mode in ("x", "y"):
            packed = _pack_regions(case, source, bounds, axis_mode=axis_mode)
            if packed is None:
                continue
            candidate, split_members, split_axis, split_cut, region_count = packed
            if not verify_feasible(case, candidate):
                continue
            key = tuple(round(float(value), 9) for value in candidate.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            records.append(
                RegionAssignmentRecord(
                    hypothesis_id,
                    axis_mode,
                    split_members,
                    split_axis,
                    split_cut,
                    region_count,
                )
            )
            if len(candidates) == count:
                return torch.stack(candidates), tuple(records)
    if not candidates:
        return source.new_empty((0, int(case.n), 4)), ()
    return torch.stack(candidates), tuple(records)


def _pack_regions(
    case: Any,
    reference: Tensor,
    bounds: tuple[float, float, float, float],
    *,
    axis_mode: str,
) -> tuple[Tensor, tuple[int, ...], str | None, int | None, int] | None:
    constrained = _place_constraint_obstacles(case, reference, bounds)
    if constrained is None:
        return None
    obstacles, occupied, _ = constrained
    occupied = occupied.to(device="cpu", dtype=torch.bool)
    free = _free_rectangles(
        bounds,
        obstacles[occupied],
        transpose=axis_mode == "y",
    )
    if not free:
        return None
    source = reference.detach().to(device="cpu", dtype=torch.float64)
    areas = case.area.detach().to(device="cpu", dtype=torch.float64)
    centers = source[:, :2] + 0.5 * source[:, 2:4]
    movable = ~occupied
    items = _island_items(case, source, movable, areas, centers)
    allocation = _allocate_items(
        items,
        free,
        boundary_bits=case.boundary_bits,
        outline=bounds,
    )
    split_members: tuple[int, ...] = ()
    split_axis: str | None = None
    split_cut: int | None = None
    if allocation is None:
        split = _best_single_split(case, items, free, areas, centers, bounds)
        if split is None:
            return None
        items, allocation, split_members, split_axis, split_cut = split

    rectangles: dict[int, tuple[float, float, float, float]] = {}
    for region, assigned in zip(free, allocation, strict=True):
        if assigned:
            _place_items(
                assigned,
                region,
                rectangles,
                areas=areas,
                centers=centers,
                axis_mode=axis_mode,
                whitespace_side="right" if axis_mode == "x" else "top",
            )
    if len(rectangles) != int(movable.sum()):
        return None
    candidate = reference.new_empty((int(case.n), 4))
    for index in torch.nonzero(movable, as_tuple=False).reshape(-1).tolist():
        candidate[index] = _xywh_tensor(
            candidate,
            rectangles[index],
            boundary_bits=case.boundary_bits[index],
            outline=bounds,
        )
    candidate[occupied.to(device=candidate.device)] = obstacles[occupied].to(
        device=candidate.device, dtype=candidate.dtype
    )
    return candidate, split_members, split_axis, split_cut, len(free)


def _island_items(
    case: Any,
    source: Tensor,
    movable: Tensor,
    areas: Tensor,
    centers: Tensor,
) -> list[_Item]:
    islands = detect_islands(source, proximity=2.0e-6)
    assigned: set[int] = set()
    items: list[_Item] = []
    for island in islands:
        members = tuple(index for index in island if bool(movable[index]))
        if not members:
            continue
        assigned.update(members)
        items.append(_item(members, areas, centers))
    for index in torch.nonzero(movable, as_tuple=False).reshape(-1).tolist():
        if index not in assigned:
            items.append(_item((index,), areas, centers))
    return items


def _best_single_split(
    case: Any,
    items: list[_Item],
    regions: list[tuple[float, float, float, float]],
    areas: Tensor,
    centers: Tensor,
    outline: tuple[float, float, float, float],
) -> (
    tuple[
        list[_Item],
        list[list[_Item]],
        tuple[int, ...],
        str,
        int,
    ]
    | None
):
    weights = torch.as_tensor(case.b2b_weight, dtype=torch.float64, device="cpu")
    choices = []
    for item_index, item in enumerate(items):
        if len(item.members) < 2:
            continue
        members = torch.tensor(item.members, dtype=torch.long)
        for axis, name in ((0, "x"), (1, "y")):
            ordered = members[
                torch.argsort(centers[members, axis], stable=True)
            ].tolist()
            for cut in range(1, len(ordered)):
                first, second = tuple(ordered[:cut]), tuple(ordered[cut:])
                cut_weight = float(weights[list(first)][:, list(second)].sum())
                balance = abs(
                    float(areas[list(first)].sum()) - float(areas[list(second)].sum())
                )
                choices.append(
                    (cut_weight, balance, item_index, axis, cut, name, first, second)
                )
    for _, _, item_index, _, cut, name, first, second in sorted(choices):
        split_items = [
            *items[:item_index],
            _item(first, areas, centers),
            _item(second, areas, centers),
            *items[item_index + 1 :],
        ]
        allocation = _allocate_items(
            split_items,
            regions,
            boundary_bits=case.boundary_bits,
            outline=outline,
        )
        if allocation is not None:
            return split_items, allocation, items[item_index].members, name, cut
    return None


__all__ = ["RegionAssignmentRecord", "obstacle_region_candidates"]
