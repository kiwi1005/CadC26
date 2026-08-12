"""Bounded exact repacking for dense disconnected groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.contact_synthesis import synthesize_contact_obligations
from hcfp.treemap import _Item, _partition
from hcfp.verify import grouping_violation, verify_feasible


Tensor = torch.Tensor
_EPS = 1.0e-9


@dataclass(frozen=True)
class ContactPatchCandidate:
    placement: Tensor
    group_index: int
    bridge_member: int
    anchor_member: int
    members: tuple[int, ...]
    side: str
    grouping_before: int
    grouping_after: int


def dense_contact_patch_candidates(
    case: Any,
    placements: Any,
    *,
    verify_case: Any | None = None,
    patch_sizes: tuple[int, ...] = (4, 8, 12, 16),
    max_candidates: int = 16,
) -> tuple[ContactPatchCandidate, ...]:
    """Re-slice a closed local patch while forcing one missing group contact."""

    boxes = torch.as_tensor(placements, dtype=torch.float64, device="cpu").clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("placements must have shape [N,4]")
    if max_candidates <= 0:
        return ()
    sizes = tuple(sorted({int(size) for size in patch_sizes if int(size) >= 2}))
    if not sizes:
        return ()

    before = grouping_violation(case, boxes)
    if before <= 0:
        return ()
    protected = _protected(case, len(boxes))
    synthesis = synthesize_contact_obligations(case, boxes)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    found: list[ContactPatchCandidate] = []
    seen: set[tuple[float, ...]] = set()

    for obligation in synthesis.obligations + synthesis.candidate_edges:
        bridge = int(obligation.bridge_member)
        anchor = int(obligation.anchor_member)
        if bridge == anchor or bool(protected[bridge]) or bool(protected[anchor]):
            continue
        distance = torch.abs(centers - 0.5 * (centers[bridge] + centers[anchor])).sum(1)
        nearest = torch.argsort(distance, stable=True).tolist()
        for size in sizes:
            seed = {bridge, anchor, *nearest[: max(0, size - 2)]}
            members = _closed_patch(boxes, seed, limit=size)
            if members is None or bool(protected[list(members)].any()):
                continue
            bounds = _bounds(boxes[list(members)])
            for side in ("left", "right", "bottom", "top"):
                candidate = _repack(
                    boxes,
                    members,
                    bridge,
                    anchor,
                    bounds,
                    side=side,
                )
                if candidate is None or not verify_feasible(
                    verify_case or case, candidate
                ):
                    continue
                after = grouping_violation(case, candidate)
                if after >= before:
                    continue
                key = tuple(round(float(value), 10) for value in candidate.reshape(-1))
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    ContactPatchCandidate(
                        candidate,
                        int(obligation.group_index),
                        bridge,
                        anchor,
                        members,
                        side,
                        before,
                        after,
                    )
                )
                if len(found) >= max_candidates:
                    return tuple(found)
    return tuple(found)


def _protected(case: Any, n: int) -> Tensor:
    fixed = torch.as_tensor(
        getattr(case, "fixed_mask", torch.zeros(n)), dtype=torch.bool
    ).reshape(-1)
    preplaced = torch.as_tensor(
        getattr(case, "preplaced_mask", torch.zeros(n)), dtype=torch.bool
    ).reshape(-1)
    mib = torch.as_tensor(
        getattr(case, "mib_membership", torch.empty((0, n))), dtype=torch.bool
    )
    mib_member = mib.any(0) if mib.numel() else torch.zeros(n, dtype=torch.bool)
    return fixed | preplaced | mib_member


def _closed_patch(
    boxes: Tensor,
    seed: set[int],
    *,
    limit: int,
) -> tuple[int, ...] | None:
    members = set(seed)
    while True:
        bounds = _bounds(boxes[list(members)])
        intersecting = {
            index for index, box in enumerate(boxes) if _intersects(box, bounds)
        }
        updated = members | intersecting
        if len(updated) > limit:
            return None
        if updated == members:
            return tuple(sorted(members))
        members = updated


def _repack(
    source: Tensor,
    members: tuple[int, ...],
    bridge: int,
    anchor: int,
    bounds: tuple[float, float, float, float],
    *,
    side: str,
) -> Tensor | None:
    patch_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    areas = source[:, 2] * source[:, 3]
    occupied = float(areas[list(members)].sum())
    if occupied > patch_area + 1.0e-7 * max(1.0, patch_area):
        return None
    centers = source[:, :2] + 0.5 * source[:, 2:4]
    pair_area = float(areas[bridge] + areas[anchor])
    pair_center = (centers[[bridge, anchor]] * areas[[bridge, anchor], None]).sum(
        0
    ) / pair_area
    items = [
        _Item(
            (bridge, anchor), pair_area, (float(pair_center[0]), float(pair_center[1]))
        )
    ]
    items.extend(
        _Item(
            (member,),
            float(areas[member]),
            (float(centers[member, 0]), float(centers[member, 1])),
        )
        for member in members
        if member not in {bridge, anchor}
    )
    whitespace = patch_area - occupied
    if whitespace > _EPS:
        items.append(_Item((), whitespace, _whitespace_center(bounds, side), True))
    rectangles: dict[int, tuple[float, float, float, float]] = {}
    _partition(
        items,
        bounds,
        rectangles,
        axis_mode="long",
        whitespace_side=side,
        depth=0,
    )
    pair = rectangles[0]
    bridge_rect, anchor_rect = _split_pair(pair, float(areas[bridge]), pair_area, side)
    candidate = source.clone()
    candidate[bridge] = _xywh(candidate, bridge_rect)
    candidate[anchor] = _xywh(candidate, anchor_rect)
    for offset, item in enumerate(items[1:], start=1):
        if item.whitespace:
            continue
        candidate[item.members[0]] = _xywh(candidate, rectangles[offset])
    return candidate


def _split_pair(
    bounds: tuple[float, float, float, float],
    bridge_area: float,
    pair_area: float,
    side: str,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    left, bottom, right, top = bounds
    ratio = bridge_area / pair_area
    if side in {"left", "right"}:
        cut = left + (right - left) * ratio
        first, second = (left, bottom, cut, top), (cut, bottom, right, top)
    else:
        cut = bottom + (top - bottom) * ratio
        first, second = (left, bottom, right, cut), (left, cut, right, top)
    return (first, second) if side in {"left", "bottom"} else (second, first)


def _xywh(template: Tensor, bounds: tuple[float, float, float, float]) -> Tensor:
    left, bottom, right, top = bounds
    return template.new_tensor((left, bottom, right - left, top - bottom))


def _bounds(boxes: Tensor) -> tuple[float, float, float, float]:
    return (
        float(boxes[:, 0].min()),
        float(boxes[:, 1].min()),
        float((boxes[:, 0] + boxes[:, 2]).max()),
        float((boxes[:, 1] + boxes[:, 3]).max()),
    )


def _intersects(box: Tensor, bounds: tuple[float, float, float, float]) -> bool:
    left, bottom, right, top = bounds
    return (
        min(float(box[0] + box[2]), right) - max(float(box[0]), left) > _EPS
        and min(float(box[1] + box[3]), top) - max(float(box[1]), bottom) > _EPS
    )


def _whitespace_center(
    bounds: tuple[float, float, float, float], side: str
) -> tuple[float, float]:
    left, bottom, right, top = bounds
    if side == "left":
        return (-float("inf"), 0.5 * (bottom + top))
    if side == "right":
        return (float("inf"), 0.5 * (bottom + top))
    if side == "bottom":
        return (0.5 * (left + right), -float("inf"))
    return (0.5 * (left + right), float("inf"))


__all__ = ["ContactPatchCandidate", "dense_contact_patch_candidates"]
