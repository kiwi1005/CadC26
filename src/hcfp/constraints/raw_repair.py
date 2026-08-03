"""Exact raw-coordinate replay for constructed soft constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from hcfp.verify import boundary_missing, grouping_violation, verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class RawConstraintRepair:
    placements: tuple[tuple[float, float, float, float], ...]
    group_edges_applied: int
    group_edges_rejected: int
    boundary_blocks_applied: int
    boundary_blocks_rejected: int


def repair_raw_constraints(
    source: Any,
    placements: Any,
    record: dict[str, object] | None,
) -> RawConstraintRepair:
    """Replay candidate provenance without ever accepting a hard regression."""

    boxes = _boxes(placements)
    details = record.get("details", {}) if isinstance(record, dict) else {}
    if not isinstance(details, dict):
        details = {}
    preplaced = _preplaced_mask(source, int(boxes.shape[0]))
    group = details.get("group", details)
    group_moves = group.get("moves", ()) if isinstance(group, dict) else ()
    boxes, group_applied, group_rejected = _repair_group_moves(
        source,
        boxes,
        group_moves,
        preplaced,
    )

    boundary = details.get("boundary", details)
    placed = boundary.get("placed", ()) if isinstance(boundary, dict) else ()
    boundary_applied = boundary_rejected = 0
    for block in placed if isinstance(placed, (tuple, list)) else ():
        index = int(block)
        if not 0 <= index < boxes.shape[0] or bool(preplaced[index]):
            boundary_rejected += 1
            continue
        candidate = _snap_boundary_block(source, boxes, index)
        before = int(boundary_missing(source, boxes).sum())
        after = int(boundary_missing(source, candidate).sum())
        if after < before and verify_feasible(source, candidate):
            boxes = candidate
            boundary_applied += 1
        elif after == before and verify_feasible(source, candidate):
            boxes = candidate
        else:
            boundary_rejected += 1

    return RawConstraintRepair(
        placements=tuple(tuple(float(value) for value in row) for row in boxes.tolist()),
        group_edges_applied=group_applied,
        group_edges_rejected=group_rejected,
        boundary_blocks_applied=boundary_applied,
        boundary_blocks_rejected=boundary_rejected,
    )


def _repair_group_moves(
    source: Any,
    boxes: Tensor,
    raw_moves: object,
    preplaced: Tensor,
) -> tuple[Tensor, int, int]:
    moves = tuple(raw_moves) if isinstance(raw_moves, (tuple, list)) else ()
    baseline = grouping_violation(source, boxes)
    baseline_feasible = verify_feasible(source, boxes)
    batch = boxes
    valid = 0
    invalid = 0
    for move in moves:
        candidate = _replay_group_move(batch, move, preplaced)
        if candidate is None:
            invalid += 1
        else:
            batch = candidate
            valid += 1
    batch_grouping = grouping_violation(source, batch)
    if valid and batch_grouping <= baseline and verify_feasible(source, batch) and (
        batch_grouping < baseline or not baseline_feasible
    ):
        return batch, valid, invalid

    working = boxes
    applied = 0
    rejected = invalid
    for move in moves:
        candidate = _replay_group_move(working, move, preplaced)
        if candidate is None:
            continue
        before = grouping_violation(source, working)
        after = grouping_violation(source, candidate)
        if after < before and verify_feasible(source, candidate):
            working = candidate
            applied += 1
        else:
            rejected += 1
    return working, applied, rejected


def _replay_group_move(
    boxes: Tensor,
    move: object,
    preplaced: Tensor,
) -> Tensor | None:
    if not isinstance(move, dict):
        return None
    try:
        members = tuple(int(value) for value in move["members"])
        anchor = int(move["anchor"])
        child = int(move["child"])
        side = str(move["side"])
    except (KeyError, TypeError, ValueError):
        return None
    n = int(boxes.shape[0])
    if (
        not members
        or any(not 0 <= index < n for index in (*members, anchor, child))
        or child not in members
        or bool(preplaced[list(members)].any())
    ):
        return None

    candidate = boxes.clone()
    ax, ay, aw, ah = (float(value) for value in candidate[anchor])
    cx, cy, cw, ch = (float(value) for value in candidate[child])
    if side == "right":
        target = (math.nextafter(ax + aw, -math.inf), cy)
    elif side == "left":
        target = (math.nextafter(ax - cw, math.inf), cy)
    elif side == "above":
        target = (cx, math.nextafter(ay + ah, -math.inf))
    elif side == "below":
        target = (cx, math.nextafter(ay - ch, math.inf))
    else:
        return None
    delta = candidate.new_tensor(target) - candidate[child, :2]
    candidate[list(members), :2] += delta
    candidate[child, :2] = candidate.new_tensor(target)
    return candidate


def _snap_boundary_block(source: Any, boxes: Tensor, block: int) -> Tensor:
    bits = _field(source, "boundary_bits")
    if bits is None:
        return boxes
    required = torch.as_tensor(bits, dtype=torch.bool, device="cpu")
    if required.shape != (boxes.shape[0], 4):
        raise ValueError("boundary_bits must have shape [N,4]")
    candidate = boxes.clone()
    left = float(candidate[:, 0].amin())
    bottom = float(candidate[:, 1].amin())
    right = float((candidate[:, 0] + candidate[:, 2]).amax())
    top = float((candidate[:, 1] + candidate[:, 3]).amax())
    if bool(required[block, 0]):
        candidate[block, 0] = left
    if bool(required[block, 1]):
        candidate[block, 0] = right - float(candidate[block, 2])
    if bool(required[block, 2]):
        candidate[block, 1] = top - float(candidate[block, 3])
    if bool(required[block, 3]):
        candidate[block, 1] = bottom
    return candidate


def _boxes(value: Any) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("placements must have shape [N,4]")
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0.0).any()):
        raise ValueError("placements must be finite with positive dimensions")
    return boxes


def _preplaced_mask(source: Any, n: int) -> Tensor:
    value = _field(source, "preplaced_mask")
    if value is None:
        constraints = _field(source, "constraints")
        if constraints is None:
            return torch.zeros(n, dtype=torch.bool)
        rules = torch.as_tensor(constraints, device="cpu")
        value = rules[:n, 1] != 0 if rules.ndim == 2 and rules.shape[1] > 1 else None
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(-1)
    if mask.numel() != n:
        raise ValueError("preplaced_mask must have shape [N]")
    return mask


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)
