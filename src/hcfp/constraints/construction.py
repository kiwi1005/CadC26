"""Exact-safe candidate variants for soft constraint construction.

The helpers are deliberately CPU-side and bounded.  They create additional
candidates; callers retain the original candidate and let the existing exact
tail decide whether a constructed variant is useful.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from hcfp.constraints.mib_shapes import resolve_mib_shapes


Tensor = torch.Tensor


@dataclass(frozen=True)
class ConstructedVariant:
    kind: str
    xywh: Tensor
    details: dict[str, object]


def construct_constraint_variants(
    case: Any,
    xywh: Tensor,
    *,
    relation_scores: Tensor | None = None,
    boundary_order_scores: Tensor | None = None,
    mib_log_aspect: Tensor | None = None,
) -> tuple[ConstructedVariant, ...]:
    """Build independent group, boundary, MIB, and combined variants."""

    boxes = _boxes(xywh)
    variants: list[ConstructedVariant] = []

    grouped, group_details = connect_groups(
        boxes,
        _field(case, "group_membership"),
        preplaced_mask=_field(case, "preplaced_mask"),
        b2b_weight=_field(case, "b2b_weight"),
        relation_scores=relation_scores,
    )
    _append_changed(variants, "group_contacts", boxes, grouped, group_details)

    boundary, boundary_details = construct_boundary_frame(
        boxes,
        _field(case, "boundary_bits"),
        preplaced_mask=_field(case, "preplaced_mask"),
        order_scores=boundary_order_scores,
    )
    _append_changed(variants, "boundary_frame", boxes, boundary, boundary_details)

    mib, mib_details = construct_mib_shapes(
        case,
        boxes,
        mib_log_aspect=mib_log_aspect,
    )
    _append_changed(variants, "mib_shapes", boxes, mib, mib_details)

    combined, combined_group = connect_groups(
        mib,
        _field(case, "group_membership"),
        preplaced_mask=_field(case, "preplaced_mask"),
        b2b_weight=_field(case, "b2b_weight"),
        relation_scores=relation_scores,
    )
    combined, combined_boundary = construct_boundary_frame(
        combined,
        _field(case, "boundary_bits"),
        preplaced_mask=_field(case, "preplaced_mask"),
        order_scores=boundary_order_scores,
    )
    _append_changed(
        variants,
        "combined",
        boxes,
        combined,
        {
            "mib": mib_details,
            "group": combined_group,
            "boundary": combined_boundary,
        },
    )
    return tuple(variants)


def connect_groups(
    xywh: Tensor,
    group_membership: Tensor | None,
    *,
    preplaced_mask: Tensor | None = None,
    b2b_weight: Tensor | None = None,
    relation_scores: Tensor | None = None,
) -> tuple[Tensor, dict[str, object]]:
    """Greedily translate movable components until group members touch."""

    boxes = _boxes(xywh)
    n = int(boxes.shape[0])
    groups = _membership(group_membership, n)
    preplaced = _mask(preplaced_mask, n)
    weights = _weights(b2b_weight, n)
    scores = _relation_scores(relation_scores, n)
    moves: list[dict[str, object]] = []
    unresolved: list[int] = []

    for group_index, row in enumerate(groups):
        members = tuple(
            int(index)
            for index in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        )
        if len(members) <= 1:
            continue
        while True:
            components = _components(boxes, members)
            if len(components) <= 1:
                break
            root = _root_component(components, preplaced)
            choice = _best_component_move(
                boxes,
                root,
                components,
                preplaced,
                weights,
                scores,
            )
            if choice is None:
                unresolved.append(group_index)
                break
            moving, delta, anchor, child, side = choice
            before = len(components)
            boxes[list(moving), :2] += delta
            after = len(_components(boxes, members))
            if after >= before:
                raise RuntimeError("accepted group move did not reduce connectivity")
            moves.append(
                {
                    "group": group_index,
                    "members": moving,
                    "anchor": anchor,
                    "child": child,
                    "side": side,
                    "dx": float(delta[0]),
                    "dy": float(delta[1]),
                    "components_before": before,
                    "components_after": after,
                }
            )
    return boxes.float(), {
        "moves": tuple(moves),
        "move_count": len(moves),
        "unresolved_groups": tuple(sorted(set(unresolved))),
    }


def construct_boundary_frame(
    xywh: Tensor,
    boundary_bits: Tensor | None,
    *,
    preplaced_mask: Tensor | None = None,
    order_scores: Tensor | None = None,
    clearance: float = 1.0e-5,
) -> tuple[Tensor, dict[str, object]]:
    """Move eligible boundary blocks into four non-overlapping outer bands."""

    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and non-negative")
    boxes = _boxes(xywh)
    n = int(boxes.shape[0])
    bits = _boundary_bits(boundary_bits, n)
    preplaced = _mask(preplaced_mask, n)
    scores = _boundary_scores(order_scores, n)
    required = bits.any(dim=1)
    if not bool(required.any()):
        return boxes.float(), {"placed": (), "skipped": (), "reason": "no_boundary"}
    if bool((required & preplaced).any()):
        skipped = tuple(
            int(i)
            for i in torch.nonzero(required, as_tuple=False).reshape(-1).tolist()
        )
        return boxes.float(), {
            "placed": (),
            "skipped": skipped,
            "reason": "preplaced_boundary_anchor",
        }

    side_names = ("left", "right", "top", "bottom")
    single: dict[str, list[int]] = {side: [] for side in side_names}
    corners: dict[tuple[str, str], int] = {}
    skipped: list[int] = []
    for block in torch.nonzero(required, as_tuple=False).reshape(-1).tolist():
        active = [side_names[index] for index in range(4) if bool(bits[block, index])]
        if len(active) == 1:
            single[active[0]].append(int(block))
            continue
        key = _corner_key(active)
        if key is not None and key not in corners:
            corners[key] = int(block)
            continue
        # Multiple blocks cannot all occupy the same exact corner.  Satisfy one
        # deterministic side and report the remaining requested sides as skipped.
        if key is not None:
            selected = min(active, key=lambda side: side_names.index(side))
            single[selected].append(int(block))
        else:
            skipped.append(int(block))

    axis = {"left": 1, "right": 1, "top": 0, "bottom": 0}
    side_index = {side: index for index, side in enumerate(side_names)}
    for side, members in single.items():
        members.sort(
            key=lambda block: (
                float(scores[block, side_index[side]]),
                float(boxes[block, axis[side]]),
                block,
            )
        )

    left = float(boxes[:, 0].amin())
    bottom = float(boxes[:, 1].amin())
    right = float((boxes[:, 0] + boxes[:, 2]).amax())
    top = float((boxes[:, 1] + boxes[:, 3]).amax())
    inner_width = max(
        right - left,
        _sum_extent(boxes, single["top"], 2, clearance),
        _sum_extent(boxes, single["bottom"], 2, clearance),
    )
    inner_height = max(
        top - bottom,
        _sum_extent(boxes, single["left"], 3, clearance),
        _sum_extent(boxes, single["right"], 3, clearance),
    )
    inner_right = left + inner_width
    inner_top = bottom + inner_height
    left_band = _max_extent(boxes, single["left"], corners, "left", 2)
    right_band = _max_extent(boxes, single["right"], corners, "right", 2)
    top_band = _max_extent(boxes, single["top"], corners, "top", 3)
    bottom_band = _max_extent(boxes, single["bottom"], corners, "bottom", 3)
    frame_left = left - left_band - clearance
    frame_right = inner_right + right_band + clearance
    frame_bottom = bottom - bottom_band - clearance
    frame_top = inner_top + top_band + clearance

    placed: list[int] = []
    cursor = bottom
    for block in single["left"]:
        boxes[block, 0] = frame_left
        boxes[block, 1] = cursor
        cursor += float(boxes[block, 3]) + clearance
        placed.append(block)
    cursor = bottom
    for block in single["right"]:
        boxes[block, 0] = frame_right - float(boxes[block, 2])
        boxes[block, 1] = cursor
        cursor += float(boxes[block, 3]) + clearance
        placed.append(block)
    cursor = left
    for block in single["top"]:
        boxes[block, 0] = cursor
        boxes[block, 1] = frame_top - float(boxes[block, 3])
        cursor += float(boxes[block, 2]) + clearance
        placed.append(block)
    cursor = left
    for block in single["bottom"]:
        boxes[block, 0] = cursor
        boxes[block, 1] = frame_bottom
        cursor += float(boxes[block, 2]) + clearance
        placed.append(block)

    for (horizontal, vertical), block in sorted(corners.items()):
        boxes[block, 0] = (
            frame_left
            if horizontal == "left"
            else frame_right - float(boxes[block, 2])
        )
        boxes[block, 1] = (
            frame_bottom
            if vertical == "bottom"
            else frame_top - float(boxes[block, 3])
        )
        placed.append(block)

    if _has_overlap(boxes):
        return _boxes(xywh).float(), {
            "placed": (),
            "skipped": tuple(sorted(set(skipped) | set(placed))),
            "reason": "constructed_overlap",
        }
    return boxes.float(), {
        "placed": tuple(sorted(set(placed))),
        "skipped": tuple(sorted(set(skipped))),
        "reason": "ok",
        "frame": (frame_left, frame_bottom, frame_right, frame_top),
    }


def construct_mib_shapes(
    case: Any,
    xywh: Tensor,
    *,
    mib_log_aspect: Tensor | None = None,
) -> tuple[Tensor, dict[str, object]]:
    """Broadcast compatible MIB shapes around the current block centers."""

    boxes = _boxes(xywh)
    n = int(boxes.shape[0])
    membership = _membership(_field(case, "mib_membership"), n)
    if not membership.numel():
        return boxes.float(), {"resolved_groups": (), "incompatible_groups": ()}
    fixed = _mask(_field(case, "fixed_mask"), n)
    preplaced = _mask(_field(case, "preplaced_mask"), n)
    hard = fixed | preplaced
    target = torch.as_tensor(
        _field(case, "target"), dtype=torch.float64, device="cpu"
    ).clone()
    if target.shape != boxes.shape:
        raise ValueError("case target must have shape [N,4]")
    if bool(hard.any()) and (
        not bool(torch.isfinite(target[hard]).all())
        or bool((target[hard, 2:4] <= 0.0).any())
    ):
        raise ValueError("hard target shapes must be finite and positive")
    hard_wh = boxes[:, 2:4].clone()
    hard_wh[hard] = target[hard, 2:4]
    proposed_wh, predicted_groups = _mib_proposed_shapes(
        _field(case, "area"),
        membership,
        boxes[:, 2:4],
        mib_log_aspect,
    )
    resolution = resolve_mib_shapes(
        _field(case, "area"),
        membership,
        proposed_wh=proposed_wh,
        hard_mask=hard,
        hard_wh=hard_wh,
    )
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    boxes[:, 2:4] = resolution.shapes.to(dtype=boxes.dtype)
    boxes[:, :2] = centers - 0.5 * boxes[:, 2:4]
    if bool(preplaced.any()):
        boxes[preplaced] = target[preplaced]
    if bool(fixed.any()):
        boxes[fixed, 2:4] = target[fixed, 2:4]
    resolved = tuple(group.group for group in resolution.groups if group.compatible)
    incompatible = tuple(group.group for group in resolution.incompatible_groups)
    return boxes.float(), {
        "resolved_groups": resolved,
        "incompatible_groups": incompatible,
        "predicted_groups": predicted_groups,
    }


def _mib_proposed_shapes(
    area: Any,
    membership: Tensor,
    current_wh: Tensor,
    mib_log_aspect: Tensor | None,
) -> tuple[Tensor, tuple[int, ...]]:
    proposed = current_wh.clone()
    if mib_log_aspect is None:
        return proposed, ()
    predictions = torch.as_tensor(
        mib_log_aspect,
        dtype=torch.float64,
        device="cpu",
    ).reshape(-1)
    if predictions.numel() != membership.shape[0]:
        raise ValueError("mib_log_aspect must have shape [M]")
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("mib_log_aspect must be finite")
    areas = torch.as_tensor(area, dtype=torch.float64, device="cpu").reshape(-1)
    if areas.numel() != membership.shape[1]:
        raise ValueError("area must have shape [N]")

    used: list[int] = []
    for group_index, row in enumerate(membership):
        members = torch.nonzero(row, as_tuple=False).reshape(-1)
        if members.numel() < 2:
            continue
        aspect = torch.exp(predictions[group_index].clamp(-4.0, 4.0))
        widths = torch.sqrt(areas[members] * aspect)
        proposed[members, 0] = widths
        proposed[members, 1] = areas[members] / widths
        used.append(group_index)
    return proposed, tuple(used)


def _best_component_move(
    boxes: Tensor,
    root: tuple[int, ...],
    components: tuple[tuple[int, ...], ...],
    preplaced: Tensor,
    weights: Tensor,
    scores: Tensor,
) -> tuple[tuple[int, ...], Tensor, int, int, str] | None:
    choices: list[
        tuple[
            tuple[float, float, int, int, int, str],
            tuple[int, ...],
            Tensor,
            int,
            int,
            str,
        ]
    ] = []
    for moving in components:
        if moving == root or bool(preplaced[list(moving)].any()):
            continue
        for anchor in root:
            for child in moving:
                for direction, target, relation_index in _contact_targets(
                    boxes[anchor], boxes[child]
                ):
                    delta = target - boxes[child, :2]
                    if _component_overlaps_outside(boxes, moving, delta):
                        continue
                    relation = float(scores[anchor, child, relation_index])
                    net = math.log1p(float(weights[anchor, child]))
                    movement = float(torch.linalg.vector_norm(delta)) * len(moving)
                    key = (
                        -(relation + net),
                        movement,
                        min(moving),
                        anchor,
                        child,
                        direction,
                    )
                    choices.append(
                        (key, moving, delta, anchor, child, direction)
                    )
    if not choices:
        return None
    _, moving, delta, anchor, child, direction = min(choices, key=lambda item: item[0])
    return moving, delta, anchor, child, direction


def _contact_targets(
    anchor: Tensor,
    child: Tensor,
) -> tuple[tuple[str, Tensor, int], ...]:
    ax, ay, aw, ah = (float(value) for value in anchor)
    _, _, cw, ch = (float(value) for value in child)
    y = min(max(float(child[1]), ay - ch + 1.0e-7), ay + ah - 1.0e-7)
    x = min(max(float(child[0]), ax - cw + 1.0e-7), ax + aw - 1.0e-7)
    return (
        ("right", child.new_tensor((ax + aw, y)), 0),
        ("left", child.new_tensor((ax - cw, y)), 1),
        ("above", child.new_tensor((x, ay + ah)), 2),
        ("below", child.new_tensor((x, ay - ch)), 3),
    )


def _components(boxes: Tensor, members: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(members)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            first = stack.pop()
            neighbors = [
                second
                for second in sorted(remaining)
                if _edge_connected(boxes[first], boxes[second])
            ]
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _root_component(
    components: tuple[tuple[int, ...], ...], preplaced: Tensor
) -> tuple[int, ...]:
    anchored = [component for component in components if bool(preplaced[list(component)].any())]
    return min(anchored or list(components), key=lambda component: (-len(component), component))


def _edge_connected(first: Tensor, second: Tensor) -> bool:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    tolerance = 1.0e-12
    return (overlap_x > 0.0 and (
        abs(ay + ah - by) <= tolerance or abs(by + bh - ay) <= tolerance
    )) or (
        overlap_y > 0.0 and (
            abs(ax + aw - bx) <= tolerance or abs(bx + bw - ax) <= tolerance
        )
    )


def _component_overlaps_outside(
    boxes: Tensor,
    component: tuple[int, ...],
    delta: Tensor,
) -> bool:
    component_index = torch.tensor(component, dtype=torch.long, device=boxes.device)
    inside = torch.zeros(int(boxes.shape[0]), dtype=torch.bool, device=boxes.device)
    inside[component_index] = True
    outside = boxes[~inside]
    if outside.numel() == 0:
        return False

    moved = boxes.index_select(0, component_index).clone()
    moved[:, :2] += delta.to(device=boxes.device, dtype=boxes.dtype)
    moved_high = moved[:, :2] + moved[:, 2:4]
    outside_high = outside[:, :2] + outside[:, 2:4]
    overlap = torch.minimum(
        moved_high[:, None],
        outside_high[None],
    ) - torch.maximum(
        moved[:, None, :2],
        outside[None, :, :2],
    )
    return bool(((overlap[..., 0] > 0.0) & (overlap[..., 1] > 0.0)).any())


def _has_overlap(boxes: Tensor) -> bool:
    for first in range(int(boxes.shape[0])):
        for second in range(first + 1, int(boxes.shape[0])):
            if _positive_overlap(boxes[first], boxes[second]):
                return True
    return False


def _positive_overlap(first: Tensor, second: Tensor) -> bool:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    return min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ah, by + bh) > max(ay, by)


def _corner_key(active: list[str]) -> tuple[str, str] | None:
    if len(active) != 2:
        return None
    horizontal = next((side for side in active if side in {"left", "right"}), None)
    vertical = next((side for side in active if side in {"top", "bottom"}), None)
    return (horizontal, vertical) if horizontal is not None and vertical is not None else None


def _sum_extent(
    boxes: Tensor,
    members: list[int],
    column: int,
    clearance: float,
) -> float:
    return sum(float(boxes[block, column]) for block in members) + clearance * max(
        0,
        len(members) - 1,
    )


def _max_extent(
    boxes: Tensor,
    members: list[int],
    corners: dict[tuple[str, str], int],
    side: str,
    column: int,
) -> float:
    blocks = list(members) + [block for sides, block in corners.items() if side in sides]
    return max((float(boxes[block, column]) for block in blocks), default=0.0)


def _append_changed(
    variants: list[ConstructedVariant],
    kind: str,
    original: Tensor,
    candidate: Tensor,
    details: dict[str, object],
) -> None:
    if not torch.equal(original, candidate):
        variants.append(ConstructedVariant(kind, candidate.float(), details))


def _boxes(value: Any) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("xywh must have shape [N,4]")
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0.0).any()):
        raise ValueError("xywh must be finite with positive dimensions")
    return boxes


def _membership(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((0, n), dtype=torch.bool)
    membership = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if membership.numel() == 0:
        return torch.zeros((0, n), dtype=torch.bool)
    if membership.ndim != 2 or membership.shape[1] != n:
        raise ValueError("membership must have shape [G,N]")
    return membership


def _mask(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(-1)
    if mask.numel() != n:
        raise ValueError("mask must have shape [N]")
    return mask


def _weights(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((n, n), dtype=torch.float64)
    weights = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if weights.shape != (n, n):
        raise ValueError("b2b_weight must have shape [N,N]")
    return weights


def _relation_scores(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((n, n, 4), dtype=torch.float64)
    scores = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if scores.shape != (n, n, 4):
        raise ValueError("relation_scores must have shape [N,N,4]")
    return scores


def _boundary_bits(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((n, 4), dtype=torch.bool)
    bits = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if bits.shape != (n, 4):
        raise ValueError("boundary_bits must have shape [N,4]")
    return bits


def _boundary_scores(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((n, 4), dtype=torch.float64)
    scores = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if scores.shape != (n, 4):
        raise ValueError("boundary order scores must have shape [N,4]")
    return scores


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)
