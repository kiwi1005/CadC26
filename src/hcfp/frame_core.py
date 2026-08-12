"""Small frame-and-core large-neighborhood challengers.

The helper in this module deliberately does not re-pack a whole floorplan.  It
keeps the current bbox as a frame, protects the rectangles that define that
frame, and tries a bounded number of rigid moves on a small active core.  The
exact verifier is the admission gate for every returned candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from hcfp.constraints.contact_tree import extract_contacts
from hcfp.geometry import centers_from_xywh
from hcfp.verify import (
    boundary_missing,
    soft_violation_normalized,
    total_hpwl,
    verify_feasible,
)


Tensor = torch.Tensor
_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class BBoxWitnesses:
    """Rectangles protected while the inner frame is being searched."""

    bounds: tuple[float, float, float, float]
    left: int
    right: int
    bottom: int
    top: int

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted({self.left, self.right, self.bottom, self.top}))


@dataclass(frozen=True)
class FrameCoreCandidate:
    """One accepted exact-safe rigid active-core move."""

    placement: Tensor
    members: tuple[int, ...]
    delta: tuple[float, float]
    strategy: str
    soft_before: float
    soft_after: float
    hpwl_before: float
    hpwl_after: float


@dataclass(frozen=True)
class FrameCoreResult:
    """Audit information and the bounded challenger population."""

    placement: Tensor
    bounds: tuple[float, float, float, float]
    witnesses: BBoxWitnesses
    active: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    candidates: tuple[FrameCoreCandidate, ...]

    @property
    def boxes(self) -> Tensor:
        return self.placement


def identify_bbox_witnesses(
    case: Any | None,
    placements: Any,
    *,
    tolerance: float = 1.0e-7,
) -> BBoxWitnesses:
    """Return one deterministic witness for each side of the current bbox.

    A required boundary block is preferred when it already lies on the
    corresponding side.  Otherwise the actual extremal rectangle is used;
    this keeps the frame invariant even when the current candidate violates a
    boundary requirement.
    """

    boxes = _boxes(placements)
    _nonnegative(tolerance, "tolerance")
    left, bottom, right, top = _bounds(boxes)
    boundary = _boundary_bits(case, len(boxes))
    side_bounds = (left, right, top, bottom)
    side_coord = (
        boxes[:, 0],
        boxes[:, 0] + boxes[:, 2],
        boxes[:, 1] + boxes[:, 3],
        boxes[:, 1],
    )
    witnesses: list[int] = []
    for side_index, (coord, target) in enumerate(zip(side_coord, side_bounds)):
        extremal = [
            index
            for index, value in enumerate(coord.tolist())
            if abs(float(value) - target) <= tolerance
        ]
        preferred = [index for index in extremal if bool(boundary[index, side_index])]
        witnesses.append(min(preferred or extremal))
    return BBoxWitnesses(
        bounds=(left, bottom, right, top),
        left=witnesses[0],
        right=witnesses[1],
        top=witnesses[2],
        bottom=witnesses[3],
    )


def frame_core_lns(
    case: Any,
    placements: Any,
    *,
    top_k: int = 8,
    max_candidates: int = 8,
    tolerance: float = 1.0e-7,
) -> FrameCoreResult:
    """Generate exact-safe dense-placement frame-and-core challengers.

    Active blocks are the union of currently violating group members,
    boundary-missing blocks, the highest weighted-degree blocks, and their
    one-hop block-to-block neighbors.  Active blocks are split into contact or
    same-group components.  For each component, this function tries contact
    alignments to nearby blocks, frame-side free slots, and a pin-median slot.

    Every accepted candidate has the original bbox, exact dimensions and
    preplaced coordinates, passes :func:`verify_feasible`, and improves either
    normalized soft violation or total HPWL.  The input is never mutated.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates < 0
    ):
        raise ValueError("max_candidates must be a non-negative integer")
    boxes = _boxes(placements)
    n = len(boxes)
    expected = _field(case, ("n", "block_count"))
    if expected is not None and int(expected) != n:
        raise ValueError(f"placements must have shape [{int(expected)}, 4]")
    witnesses = identify_bbox_witnesses(case, boxes, tolerance=tolerance)
    protected = set(witnesses.indices)
    preplaced = _preplaced(case, n)
    protected.update(
        int(i) for i in torch.nonzero(preplaced, as_tuple=False).reshape(-1).tolist()
    )
    active = _active_blocks(case, boxes, top_k=top_k)
    active.difference_update(protected)
    components = _active_components(case, boxes, active, tolerance=tolerance)
    if not components or not max_candidates:
        return FrameCoreResult(
            boxes.clone(),
            witnesses.bounds,
            witnesses,
            tuple(sorted(active)),
            components,
            (),
        )

    soft_before = float(soft_violation_normalized(case, boxes).total)
    hpwl_before = float(total_hpwl(case, boxes))
    proposed = _proposals(
        case,
        boxes,
        components,
        witnesses.bounds,
        tolerance=tolerance,
    )
    accepted: list[FrameCoreCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for members, delta, strategy in proposed:
        moved = boxes.clone()
        moved[list(members), :2] += moved.new_tensor(delta)
        if not _safe_move(
            case,
            boxes,
            moved,
            witnesses.bounds,
            preplaced,
            protected,
            tolerance=tolerance,
        ):
            continue
        soft_after = float(soft_violation_normalized(case, moved).total)
        hpwl_after = float(total_hpwl(case, moved))
        if not (soft_after < soft_before - 1.0e-9 or hpwl_after < hpwl_before - 1.0e-9):
            continue
        key = tuple(round(float(value), 12) for value in moved.reshape(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            FrameCoreCandidate(
                moved,
                members,
                (float(delta[0]), float(delta[1])),
                strategy,
                soft_before,
                soft_after,
                hpwl_before,
                hpwl_after,
            )
        )
        if len(accepted) >= max_candidates:
            break
    return FrameCoreResult(
        boxes.clone(),
        witnesses.bounds,
        witnesses,
        tuple(sorted(active)),
        components,
        tuple(accepted),
    )


def frame_core_candidates(
    case: Any,
    placements: Any,
    **kwargs: Any,
) -> tuple[Tensor, ...]:
    """Convenience API returning only accepted placement tensors."""

    return tuple(
        candidate.placement
        for candidate in frame_core_lns(case, placements, **kwargs).candidates
    )


generate_frame_core_candidates = frame_core_candidates


def _active_blocks(case: Any, boxes: Tensor, *, top_k: int) -> set[int]:
    n = len(boxes)
    active: set[int] = set()
    missing = boundary_missing(case, boxes)
    active.update(
        int(i) for i in torch.nonzero(missing != 0, as_tuple=False).reshape(-1).tolist()
    )
    groups = _groups(case, n)
    for row in groups:
        members = [
            int(i) for i in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        ]
        if len(members) > 1 and _group_components(boxes, members) > 1:
            active.update(members)
    weights = _b2b(case, n)
    if top_k:
        degree = weights.sum(dim=1)
        limit = min(top_k, n)
        active.update(
            int(i)
            for i in torch.argsort(degree, descending=True, stable=True)[
                :limit
            ].tolist()
        )
    if active and bool(weights.numel()):
        neighbors = torch.zeros(n, dtype=torch.bool)
        for index in active:
            neighbors |= weights[index] > 0
        active.update(
            int(i)
            for i in torch.nonzero(neighbors, as_tuple=False).reshape(-1).tolist()
        )
    return active


def _active_components(
    case: Any,
    boxes: Tensor,
    active: set[int],
    *,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    if not active:
        return ()
    n = len(boxes)
    adjacency = {index: set() for index in active}
    contacts = extract_contacts(
        boxes,
        net_weight=_b2b(case, n),
        tolerance=max(tolerance, 1.0e-9),
    )
    for contact in contacts:
        if contact.first in active and contact.second in active:
            adjacency[contact.first].add(contact.second)
            adjacency[contact.second].add(contact.first)
    for row in _groups(case, n):
        members = [
            index
            for index in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
            if int(index) in active
        ]
        for first, second in zip(members, members[1:]):
            adjacency[int(first)].add(int(second))
            adjacency[int(second)].add(int(first))
    result: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in sorted(active):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for other in sorted(adjacency[current], reverse=True):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(tuple(sorted(members)))
    return tuple(sorted(result))


def _proposals(
    case: Any,
    boxes: Tensor,
    components: tuple[tuple[int, ...], ...],
    bounds: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> tuple[tuple[tuple[int, ...], tuple[float, float], str], ...]:
    proposals: list[tuple[tuple[int, ...], tuple[float, float], str]] = []
    all_indices = set(range(len(boxes)))
    groups = _groups(case, len(boxes))
    for members in components:
        member_set = set(members)
        anchors = sorted(all_indices - member_set)
        # Same-group anchors are the most likely to remove a grouping defect.
        same_group = [
            index
            for index in anchors
            if any(
                bool(row[index]) and any(bool(row[m]) for m in members)
                for row in groups
            )
        ]
        ordered_anchors = same_group + [
            index for index in anchors if index not in set(same_group)
        ]
        for anchor in ordered_anchors:
            proposals.extend(_contact_proposals(members, anchor, boxes, bounds))
        proposals.extend(_frame_proposals(members, boxes, bounds))
        pin_delta = _pin_delta(case, boxes, members, bounds)
        if pin_delta is not None:
            proposals.append((members, pin_delta, "pin_free_slot"))
    # Try short moves first.  It makes the bounded output deterministic and
    # tends to preserve the already-good local geometry.
    proposals.sort(key=lambda item: (math.hypot(*item[1]), item[2], item[0], item[1]))
    return tuple(proposals)


def _contact_proposals(
    members: tuple[int, ...],
    anchor: int,
    boxes: Tensor,
    bounds: tuple[float, float, float, float],
) -> list[tuple[tuple[int, ...], tuple[float, float], str]]:
    c_left, c_bottom, c_right, c_top = _member_bounds(boxes, members)
    a_left = float(boxes[anchor, 0])
    a_bottom = float(boxes[anchor, 1])
    a_right = float(boxes[anchor, 0] + boxes[anchor, 2])
    a_top = float(boxes[anchor, 1] + boxes[anchor, 3])
    c_center_x = 0.5 * (c_left + c_right)
    c_center_y = 0.5 * (c_bottom + c_top)
    a_center_x = 0.5 * (a_left + a_right)
    a_center_y = 0.5 * (a_bottom + a_top)
    frame_left, frame_bottom, frame_right, frame_top = bounds
    proposals: list[tuple[tuple[int, ...], tuple[float, float], str]] = []
    targets = (
        (a_left - c_right, a_center_y - c_center_y, "contact_left"),
        (a_right - c_left, a_center_y - c_center_y, "contact_right"),
        (a_center_x - c_center_x, a_bottom - c_top, "contact_bottom"),
        (a_center_x - c_center_x, a_top - c_bottom, "contact_top"),
    )
    for dx, dy, strategy in targets:
        # A rigid component cannot be shifted outside the protected frame.
        if c_left + dx < frame_left - 1.0e-8 or c_right + dx > frame_right + 1.0e-8:
            continue
        if c_bottom + dy < frame_bottom - 1.0e-8 or c_top + dy > frame_top + 1.0e-8:
            continue
        if abs(dx) + abs(dy) > 1.0e-9:
            proposals.append((members, (dx, dy), strategy))
    return proposals


def _frame_proposals(
    members: tuple[int, ...],
    boxes: Tensor,
    bounds: tuple[float, float, float, float],
) -> list[tuple[tuple[int, ...], tuple[float, float], str]]:
    left, bottom, right, top = _member_bounds(boxes, members)
    frame_left, frame_bottom, frame_right, frame_top = bounds
    candidates = (
        (frame_left - left, 0.0, "free_left"),
        (frame_right - right, 0.0, "free_right"),
        (0.0, frame_bottom - bottom, "free_bottom"),
        (0.0, frame_top - top, "free_top"),
    )
    return [
        (members, (dx, dy), strategy)
        for dx, dy, strategy in candidates
        if abs(dx) + abs(dy) > 1.0e-9
    ]


def _safe_move(
    case: Any,
    source: Tensor,
    moved: Tensor,
    bounds: tuple[float, float, float, float],
    preplaced: Tensor,
    protected: set[int],
    *,
    tolerance: float,
) -> bool:
    if not bool(torch.equal(source[:, 2:4], moved[:, 2:4])):
        return False
    if bool(preplaced.any()) and not bool(
        torch.equal(source[preplaced], moved[preplaced])
    ):
        return False
    protected_tensor = torch.tensor(sorted(protected), dtype=torch.long)
    if protected_tensor.numel() and not bool(
        torch.equal(source[protected_tensor], moved[protected_tensor])
    ):
        return False
    left, bottom, right, top = _bounds(moved)
    if max(
        abs(left - bounds[0]),
        abs(bottom - bounds[1]),
        abs(right - bounds[2]),
        abs(top - bounds[3]),
    ) > max(tolerance, 1.0e-6):
        return False
    if not verify_feasible(case, moved):
        return False
    return True


def _pin_delta(
    case: Any,
    boxes: Tensor,
    members: tuple[int, ...],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    edges = _field(case, ("p2b_edges", "p2b_connectivity"))
    pins = _field(case, ("pins", "pins_pos", "pin_positions"))
    if edges is None or pins is None:
        return None
    edge_tensor = torch.as_tensor(edges, dtype=torch.float64)
    pin_tensor = torch.as_tensor(pins, dtype=torch.float64)
    if not edge_tensor.numel() or edge_tensor.ndim != 2 or edge_tensor.shape[1] < 3:
        return None
    selected = []
    member_set = set(members)
    for row in edge_tensor.tolist():
        pin, block, weight = row[:3]
        if int(block) in member_set and int(pin) >= 0 and float(weight) > 0:
            selected.append((int(pin), float(weight)))
    if not selected:
        return None
    center = centers_from_xywh(boxes[list(members)]).mean(dim=0)
    offsets: list[tuple[float, float]] = []
    weights: list[float] = []
    for pin, weight in selected:
        if pin >= len(pin_tensor):
            continue
        offsets.append(
            tuple(float(value) for value in (pin_tensor[pin] - center).tolist())
        )
        weights.append(weight)
    if not offsets:
        return None
    dx = _weighted_median([value[0] for value in offsets], weights)
    dy = _weighted_median([value[1] for value in offsets], weights)
    c_left, c_bottom, c_right, c_top = _member_bounds(boxes, members)
    dx = max(bounds[0] - c_left, min(bounds[2] - c_right, dx))
    dy = max(bounds[1] - c_bottom, min(bounds[3] - c_top, dy))
    if abs(dx) + abs(dy) <= 1.0e-9:
        return None
    return float(dx), float(dy)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    total = sum(weights[index] for index in order)
    cumulative = 0.0
    for index in order:
        cumulative += weights[index]
        if cumulative >= 0.5 * total:
            return values[index]
    return values[order[-1]]


def _group_components(
    boxes: Tensor, members: list[int], tolerance: float = 1.0e-9
) -> int:
    adjacency = {member: set() for member in members}
    contacts = extract_contacts(boxes, tolerance=tolerance)
    member_set = set(members)
    for contact in contacts:
        if contact.first in member_set and contact.second in member_set:
            adjacency[contact.first].add(contact.second)
            adjacency[contact.second].add(contact.first)
    seen: set[int] = set()
    count = 0
    for start in members:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return count


def _member_bounds(
    boxes: Tensor, members: tuple[int, ...]
) -> tuple[float, float, float, float]:
    selected = boxes[list(members)]
    return (
        float(selected[:, 0].min()),
        float(selected[:, 1].min()),
        float((selected[:, 0] + selected[:, 2]).max()),
        float((selected[:, 1] + selected[:, 3]).max()),
    )


def _bounds(boxes: Tensor) -> tuple[float, float, float, float]:
    return (
        float(boxes[:, 0].min()),
        float(boxes[:, 1].min()),
        float((boxes[:, 0] + boxes[:, 2]).max()),
        float((boxes[:, 1] + boxes[:, 3]).max()),
    )


def _boxes(value: Any) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
        raise ValueError("placements must have shape [N,4] and contain blocks")
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:] <= 0).any()):
        raise ValueError("placements must contain finite positive dimensions")
    return boxes


def _field(source: Any | None, names: tuple[str, ...]) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return next((source[name] for name in names if name in source), None)
    return next(
        (getattr(source, name) for name in names if hasattr(source, name)), None
    )


def _groups(case: Any, n: int) -> Tensor:
    value = _field(case, ("group_membership", "groups", "cluster_membership"))
    if value is None:
        return torch.empty((0, n), dtype=torch.bool)
    groups = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if groups.ndim != 2 or groups.shape[1] != n:
        raise ValueError("group membership must have shape [G,N]")
    return groups


def _boundary_bits(case: Any | None, n: int) -> Tensor:
    value = _field(case, ("boundary_bits", "boundary_mask", "boundary_codes"))
    if value is None:
        return torch.zeros((n, 4), dtype=torch.bool)
    bits = torch.as_tensor(value, device="cpu")
    if bits.ndim == 2 and bits.shape == (n, 4):
        return bits.to(dtype=torch.bool)
    codes = bits.reshape(-1).to(dtype=torch.long)
    if codes.numel() != n:
        raise ValueError("boundary bits must have shape [N,4] or [N]")
    return torch.stack([(codes & bit) != 0 for bit in (1, 2, 4, 8)], dim=1)


def _b2b(case: Any, n: int) -> Tensor:
    value = _field(case, ("b2b_weight", "b2b_weights", "net_weight_matrix"))
    if value is not None:
        weight = torch.as_tensor(value, dtype=torch.float64, device="cpu")
        if weight.shape != (n, n):
            raise ValueError("b2b_weight must have shape [N,N]")
        return weight
    edges = _field(case, ("b2b_connectivity",))
    dense = torch.zeros((n, n), dtype=torch.float64)
    if edges is None:
        return dense
    for row in torch.as_tensor(edges, dtype=torch.float64).reshape(-1, 3).tolist():
        i, j, weight = (int(row[0]), int(row[1]), float(row[2]))
        if i >= 0 and j >= 0 and i < n and j < n and weight > 0:
            dense[i, j] += weight
            dense[j, i] += weight
    return dense


def _preplaced(case: Any, n: int) -> Tensor:
    value = _field(case, ("preplaced_mask", "is_preplaced"))
    if value is None:
        constraints = _field(case, ("constraints", "target_constraints"))
        if constraints is not None:
            rows = torch.as_tensor(constraints)
            if rows.ndim == 2 and rows.shape[0] >= n and rows.shape[1] > 1:
                value = rows[:n, 1] != 0
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(-1)
    if mask.numel() != n:
        raise ValueError("preplaced_mask must have shape [N]")
    return mask


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")
