"""Contact C0-C2 corruptions with decoder-verifiable inverse actions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import permutations
from typing import Any

import torch

from hcfp.constraints.contact_tree import Contact, extract_contacts
from hcfp.repair.decoders.contact import decode_contact_action
from hcfp.repair.decoders.packing import closed_patch, strip_reslice
from hcfp.repair.schema import ExpertKind, RepairAction
from hcfp.verify import grouping_violation, verify_feasible


_AWAY = {
    "LEFT": (-1.0, 0.0),
    "RIGHT": (1.0, 0.0),
    "TOP": (0.0, 1.0),
    "BOTTOM": (0.0, -1.0),
}
_TARGET_RELATION = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "TOP": "BOTTOM",
    "BOTTOM": "TOP",
}
_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "TOP": "BOTTOM", "BOTTOM": "TOP"}
_RATIOS = {"C0": (1.0e-4, 5.0e-4, 1.0e-3), "C1": (0.05, 0.10, 0.25)}


@dataclass(frozen=True)
class ContactCorruption:
    kind: str
    placement: torch.Tensor
    inverse_action: RepairAction
    debt_before: int
    debt_after: int
    decoded_debt: int


def generate_contact_corruptions(
    case: Any,
    placement: Any,
    *,
    verify_case: Any | None = None,
    kinds: tuple[str, ...] = ("C0", "C1"),
) -> tuple[ContactCorruption, ...]:
    boxes = torch.as_tensor(placement, dtype=torch.float64, device="cpu").clone()
    verifier = verify_case or case
    if not verify_feasible(verifier, boxes):
        return ()
    requested = tuple(dict.fromkeys(kind.upper() for kind in kinds))
    if any(kind not in {"C0", "C1", "C2"} for kind in requested):
        raise ValueError("Contact corruption kind must be C0, C1, or C2")
    before = grouping_violation(case, boxes)
    contacts = extract_contacts(boxes, tolerance=0.0)
    preplaced = torch.as_tensor(case.preplaced_mask, dtype=torch.bool, device="cpu")
    choices = _choices(case, contacts, preplaced)
    found: list[ContactCorruption] = []
    for kind in requested:
        if kind == "C2":
            corruption = _generate_c2(case, boxes, verifier, before, contacts)
            if corruption is not None:
                found.append(corruption)
            continue
        for group_index, degree, contact, target, anchor, toward_side in choices:
            if kind == "C0" and degree[target] != 1:
                continue
            relation = _TARGET_RELATION[toward_side]
            scale = min(float(boxes[target, 2]), float(boxes[target, 3]))
            for ratio in _RATIOS[kind]:
                gap = max(1.0e-3, scale * ratio)
                for moving, dx, dy in _mutations(boxes, target, anchor, relation, gap):
                    if bool(preplaced[list(moving)].any()):
                        continue
                    candidate = boxes.clone()
                    candidate[list(moving), 0] += dx
                    candidate[list(moving), 1] += dy
                    if not verify_feasible(verifier, candidate):
                        continue
                    after = grouping_violation(case, candidate)
                    if after <= before:
                        continue
                    decoded = None
                    action = None
                    for patch_budget in (2,) if kind == "C0" else (2, 4, 8, 16):
                        proposed = RepairAction(
                            expert=ExpertKind.CONTACT,
                            obligation_id=f"contact-group:{group_index}",
                            target_ids=(target,),
                            anchor_ids=(anchor,),
                            relation=relation,
                            patch_budget=patch_budget,
                            corruption_id=None,
                        )
                        proposed = replace(
                            proposed,
                            corruption_id=_corruption_id(
                                kind, proposed, gap, moving, dx, dy
                            ),
                        )
                        result = decode_contact_action(
                            case,
                            candidate,
                            proposed,
                            verify_case=verifier,
                        )
                        if result.succeeded and result.debt_after is not None:
                            action, decoded = proposed, result
                            break
                    if action is None or decoded is None:
                        continue
                    found.append(
                        ContactCorruption(
                            kind,
                            candidate,
                            action,
                            before,
                            after,
                            decoded.debt_after,
                        )
                    )
                    break
                if found and found[-1].kind == kind:
                    break
            if found and found[-1].kind == kind:
                break
    return tuple(found)


def contact_c2_eligible(case: Any, placement: Any) -> bool:
    boxes = torch.as_tensor(placement, dtype=torch.float64, device="cpu")
    contacts = extract_contacts(boxes, tolerance=0.0)
    return bool(_c2_choices(case, boxes, contacts))


def _generate_c2(case, boxes, verifier, before, contacts) -> ContactCorruption | None:
    for group_index, target, anchor, members in _c2_choices(case, boxes, contacts):
        for order in permutations(members):
            target_slot, anchor_slot = order.index(target), order.index(anchor)
            if abs(target_slot - anchor_slot) != 1:
                continue
            first = target if target_slot < anchor_slot else anchor
            for axis in ("x", "y"):
                relation = (
                    ("LEFT" if target_slot < anchor_slot else "RIGHT")
                    if axis == "x"
                    else ("BOTTOM" if target_slot < anchor_slot else "TOP")
                )
                candidate = strip_reslice(
                    boxes,
                    members,
                    order,
                    axis=axis,
                    whitespace_after=first,
                )
                if candidate is None or not verify_feasible(verifier, candidate):
                    continue
                after = grouping_violation(case, candidate)
                if after <= before:
                    continue
                for patch_budget in (2, 4, 8, 16):
                    action = RepairAction(
                        ExpertKind.CONTACT,
                        f"contact-group:{group_index}",
                        (target,),
                        (anchor,),
                        relation,
                        patch_budget=patch_budget,
                    )
                    action = replace(
                        action,
                        corruption_id=_corruption_id(
                            "C2",
                            action,
                            0.0,
                            order,
                            float(axis == "y"),
                            0.0,
                        ),
                    )
                    decoded = decode_contact_action(
                        case,
                        candidate,
                        action,
                        verify_case=verifier,
                    )
                    if decoded.succeeded and decoded.debt_after is not None:
                        return ContactCorruption(
                            "C2",
                            candidate,
                            action,
                            before,
                            after,
                            decoded.debt_after,
                        )
    return None


def _c2_choices(case, boxes, contacts):
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool, device="cpu")
    preplaced = torch.as_tensor(case.preplaced_mask, dtype=torch.bool, device="cpu")
    fixed = torch.as_tensor(case.fixed_mask, dtype=torch.bool, device="cpu")
    mib = torch.as_tensor(case.mib_membership, dtype=torch.bool, device="cpu")
    shape_locked = preplaced | fixed | (mib.any(0) if mib.numel() else False)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    bridges = _group_bridges(groups, contacts)
    choices = set()
    for group_index, row in enumerate(groups):
        group_members = set(torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        for contact in contacts:
            if (
                contact.first not in group_members
                or contact.second not in group_members
            ):
                continue
            edge = (
                group_index,
                min(contact.first, contact.second),
                max(contact.first, contact.second),
            )
            if edge not in bridges:
                continue
            for target, anchor in (
                (contact.first, contact.second),
                (contact.second, contact.first),
            ):
                distance = torch.abs(
                    centers - 0.5 * (centers[target] + centers[anchor])
                ).sum(1)
                nearest = [
                    index
                    for index in torch.argsort(distance, stable=True).tolist()
                    if index not in {target, anchor}
                ]
                for extra in range(3):
                    seeds = (target, anchor, *nearest[:extra])
                    members = closed_patch(boxes, seeds, max_blocks=4)
                    if members is None or bool(shape_locked[list(members)].any()):
                        continue
                    order = (
                        target,
                        anchor,
                        *(m for m in members if m not in {target, anchor}),
                    )
                    if any(
                        strip_reslice(
                            boxes,
                            members,
                            order,
                            axis=axis,
                            whitespace_after=target,
                        )
                        is not None
                        for axis in ("x", "y")
                    ):
                        choices.add((group_index, target, anchor, members))
    return tuple(sorted(choices))


def _group_bridges(groups: torch.Tensor, contacts: tuple[Contact, ...]):
    bridges = set()
    for group_index, row in enumerate(groups):
        members = set(torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        edges = {
            (min(contact.first, contact.second), max(contact.first, contact.second))
            for contact in contacts
            if contact.first in members and contact.second in members
        }
        for edge in edges:
            adjacency = {member: set() for member in members}
            for first, second in edges - {edge}:
                adjacency[first].add(second)
                adjacency[second].add(first)
            if not _connected(members, adjacency):
                bridges.add((group_index, *edge))
    return bridges


def _choices(case: Any, contacts: tuple[Contact, ...], preplaced: torch.Tensor):
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool, device="cpu")
    choices = []
    for group_index, row in enumerate(groups):
        members = set(torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        group_contacts = tuple(
            edge
            for edge in contacts
            if edge.first in members and edge.second in members
        )
        degree = {member: 0 for member in members}
        adjacency = {member: set() for member in members}
        for edge in group_contacts:
            degree[edge.first] += 1
            degree[edge.second] += 1
            adjacency[edge.first].add(edge.second)
            adjacency[edge.second].add(edge.first)
        if not _connected(members, adjacency):
            continue
        for edge in group_contacts:
            endpoints = (
                (edge.first, edge.second, edge.first_side),
                (edge.second, edge.first, edge.second_side),
            )
            for target, anchor, toward_side in endpoints:
                if not bool(preplaced[target]):
                    choices.append(
                        (group_index, degree, edge, target, anchor, toward_side)
                    )
    return tuple(
        sorted(
            choices,
            key=lambda item: (
                item[0],
                item[1][item[3]],
                item[3],
                item[4],
                item[5],
            ),
        )
    )


def _connected(members: set[int], adjacency: dict[int, set[int]]) -> bool:
    if len(members) < 2:
        return False
    found = {min(members)}
    stack = list(found)
    while stack:
        new = adjacency[stack.pop()] - found
        found.update(new)
        stack.extend(new)
    return found == members


def _moving_sets(
    boxes: torch.Tensor,
    target: int,
    relation: str,
    gap: float,
) -> tuple[tuple[int, ...], ...]:
    target_box = boxes[target]
    if relation == "RIGHT":
        selected = torch.nonzero(boxes[:, 0] >= target_box[0], as_tuple=False)
    elif relation == "LEFT":
        selected = torch.nonzero(
            boxes[:, 0] + boxes[:, 2] <= target_box[0] + target_box[2],
            as_tuple=False,
        )
    elif relation == "TOP":
        selected = torch.nonzero(boxes[:, 1] >= target_box[1], as_tuple=False)
    else:
        selected = torch.nonzero(
            boxes[:, 1] + boxes[:, 3] <= target_box[1] + target_box[3],
            as_tuple=False,
        )
    half_plane = tuple(sorted(selected.reshape(-1).tolist()))
    direction = _AWAY[relation]
    closure = _translation_closure(
        boxes,
        target,
        direction[0] * gap,
        direction[1] * gap,
    )
    return tuple(dict.fromkeys(((target,), closure, half_plane)))


def _mutations(
    boxes: torch.Tensor,
    target: int,
    anchor: int,
    relation: str,
    gap: float,
) -> tuple[tuple[tuple[int, ...], float, float], ...]:
    direction = _AWAY[relation]
    target_moves = tuple(
        (moving, direction[0] * gap, direction[1] * gap)
        for moving in _moving_sets(boxes, target, relation, gap)
    )
    opposite = _OPPOSITE[relation]
    anchor_direction = _AWAY[opposite]
    anchor_moves = tuple(
        (moving, anchor_direction[0] * gap, anchor_direction[1] * gap)
        for moving in _moving_sets(boxes, anchor, opposite, gap)
    )
    return tuple(dict.fromkeys(target_moves + anchor_moves))


def _translation_closure(
    boxes: torch.Tensor,
    target: int,
    dx: float,
    dy: float,
) -> tuple[int, ...]:
    moving = {target}
    while True:
        candidate = boxes.clone()
        candidate[list(moving), 0] += dx
        candidate[list(moving), 1] += dy
        blockers = set()
        for first in moving:
            for second in range(len(boxes)):
                if second in moving:
                    continue
                x_overlap = min(
                    float(candidate[first, 0] + candidate[first, 2]),
                    float(candidate[second, 0] + candidate[second, 2]),
                ) - max(float(candidate[first, 0]), float(candidate[second, 0]))
                y_overlap = min(
                    float(candidate[first, 1] + candidate[first, 3]),
                    float(candidate[second, 1] + candidate[second, 3]),
                ) - max(float(candidate[first, 1]), float(candidate[second, 1]))
                if x_overlap > 1.0e-9 and y_overlap > 1.0e-9:
                    blockers.add(second)
        if not blockers:
            return tuple(sorted(moving))
        moving.update(blockers)


def _corruption_id(
    kind: str,
    action: RepairAction,
    gap: float,
    moving: tuple[int, ...],
    dx: float,
    dy: float,
) -> str:
    payload = (
        kind,
        action.obligation_id,
        action.target_ids,
        action.anchor_ids,
        action.relation,
        action.patch_budget,
        moving,
        round(dx, 12),
        round(dy, 12),
        round(gap, 12),
    )
    return f"contact-{kind.lower()}:{hashlib.sha256(repr(payload).encode()).hexdigest()[:16]}"
