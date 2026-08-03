"""Boundary virtual-side slot construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Side = Literal["left", "right", "top", "bottom"]
SIDE_ORDER: tuple[Side, ...] = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class BoundarySlot:
    block: int
    side: Side
    primary: float
    secondary: float
    ordinal: int


@dataclass(frozen=True)
class SideEqualityConstraint:
    block: int
    block_side: Side
    virtual_side: Side


@dataclass(frozen=True)
class SideOrderConstraint:
    before: int
    after: int
    side: Side


@dataclass(frozen=True)
class BoundarySlotConstraints:
    slots: tuple[BoundarySlot, ...]
    equalities: tuple[SideEqualityConstraint, ...]
    orders: tuple[SideOrderConstraint, ...]


def construct_boundary_slots(boundary_bits: torch.Tensor, boxes: torch.Tensor) -> BoundarySlotConstraints:
    """Create deterministic virtual-side equality and order constraints."""

    bits = torch.as_tensor(boundary_bits, dtype=torch.bool, device="cpu")
    rects = torch.as_tensor(boxes, dtype=torch.float64, device="cpu")
    if bits.ndim != 2 or bits.shape[1] != 4:
        raise ValueError("boundary_bits must have shape [N,4] in left/right/top/bottom order")
    if rects.shape != (bits.shape[0], 4):
        raise ValueError("boxes must have shape [N,4]")
    if not bool(torch.isfinite(rects).all()) or bool((rects[:, 2:] <= 0).any()):
        raise ValueError("boxes must be finite with positive dimensions")

    slots_by_side: dict[Side, list[tuple[float, float, int]]] = {side: [] for side in SIDE_ORDER}
    equalities: list[SideEqualityConstraint] = []
    for block in range(int(bits.shape[0])):
        for bit_index, side in enumerate(SIDE_ORDER):
            if bool(bits[block, bit_index]):
                primary, secondary = _slot_key(rects[block], side)
                slots_by_side[side].append((primary, secondary, block))
                equalities.append(SideEqualityConstraint(block, side, side))

    slots: list[BoundarySlot] = []
    orders: list[SideOrderConstraint] = []
    for side in SIDE_ORDER:
        ordered = sorted(slots_by_side[side], key=lambda item: (item[0], item[1], item[2]))
        previous: BoundarySlot | None = None
        for ordinal, (primary, secondary, block) in enumerate(ordered):
            slot = BoundarySlot(block=block, side=side, primary=primary, secondary=secondary, ordinal=ordinal)
            slots.append(slot)
            if previous is not None:
                orders.append(SideOrderConstraint(previous.block, block, side))
            previous = slot

    return BoundarySlotConstraints(tuple(slots), tuple(equalities), tuple(orders))


def _slot_key(box: torch.Tensor, side: Side) -> tuple[float, float]:
    x, y, width, height = (float(v) for v in box.tolist())
    if side in ("left", "right"):
        return y + 0.5 * height, x
    if side in ("top", "bottom"):
        return x + 0.5 * width, y
    raise ValueError(f"unknown side: {side}")
