"""Small public packing primitives used by repair decoders."""

from __future__ import annotations

from typing import Any

import torch


def closed_patch(
    placement: Any,
    seeds: tuple[int, ...],
    *,
    max_blocks: int,
) -> tuple[int, ...] | None:
    boxes = torch.as_tensor(placement, dtype=torch.float64, device="cpu")
    members = set(int(index) for index in seeds)
    if not members or max_blocks < len(members):
        return None
    while True:
        left = float(boxes[list(members), 0].min())
        bottom = float(boxes[list(members), 1].min())
        right = float((boxes[list(members), 0] + boxes[list(members), 2]).max())
        top = float((boxes[list(members), 1] + boxes[list(members), 3]).max())
        intersecting = {
            index
            for index, box in enumerate(boxes)
            if min(float(box[0] + box[2]), right) - max(float(box[0]), left) > 1.0e-9
            and min(float(box[1] + box[3]), top) - max(float(box[1]), bottom) > 1.0e-9
        }
        updated = members | intersecting
        if len(updated) > max_blocks:
            return None
        if updated == members:
            return tuple(sorted(members))
        members = updated


def strip_reslice(
    placement: Any,
    members: tuple[int, ...],
    order: tuple[int, ...],
    *,
    axis: str,
    whitespace_after: int,
) -> torch.Tensor | None:
    boxes = torch.as_tensor(placement, dtype=torch.float64, device="cpu").clone()
    if set(order) != set(members) or axis not in {"x", "y"}:
        raise ValueError(
            "order must contain the patch members exactly and axis must be x/y"
        )
    patch = boxes[list(members)]
    left = float(patch[:, 0].min())
    bottom = float(patch[:, 1].min())
    right = float((patch[:, 0] + patch[:, 2]).max())
    top = float((patch[:, 1] + patch[:, 3]).max())
    patch_area = (right - left) * (top - bottom)
    areas = boxes[:, 2] * boxes[:, 3]
    whitespace = patch_area - float(areas[list(members)].sum())
    if whitespace <= 1.0e-7 * max(1.0, patch_area):
        return None
    span = top - bottom if axis == "x" else right - left
    cursor = left if axis == "x" else bottom
    whitespace_span = whitespace / span
    for member in order:
        extent = float(areas[member]) / span
        if axis == "x":
            boxes[member] = boxes.new_tensor((cursor, bottom, extent, top - bottom))
        else:
            boxes[member] = boxes.new_tensor((left, cursor, right - left, extent))
        cursor += extent
        if member == whitespace_after:
            cursor += whitespace_span
    return boxes


__all__ = ["closed_patch", "strip_reslice"]
