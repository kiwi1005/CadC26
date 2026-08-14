"""Contact C0/C1 corruptions with decoder-verifiable inverse actions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import torch

from hcfp.constraints.contact_tree import Contact, extract_contacts
from hcfp.repair.decoders.contact import decode_contact_action
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
    if any(kind not in _RATIOS for kind in requested):
        raise ValueError("Contact corruption kind must be C0 or C1")
    before = grouping_violation(case, boxes)
    contacts = extract_contacts(boxes, tolerance=0.0)
    preplaced = torch.as_tensor(case.preplaced_mask, dtype=torch.bool, device="cpu")
    choices = _choices(case, contacts, preplaced)
    found: list[ContactCorruption] = []
    for kind in requested:
        for group_index, degree, contact, target, anchor, toward_side in choices:
            if kind == "C0" and degree[target] != 1:
                continue
            relation = _TARGET_RELATION[toward_side]
            scale = min(float(boxes[target, 2]), float(boxes[target, 3]))
            for ratio in _RATIOS[kind]:
                gap = max(1.0e-3, scale * ratio)
                for moving, dx, dy in _mutations(
                    boxes, target, anchor, relation, gap
                ):
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
                    for patch_budget in ((2,) if kind == "C0" else (2, 4, 8, 16)):
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


def _choices(case: Any, contacts: tuple[Contact, ...], preplaced: torch.Tensor):
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool, device="cpu")
    choices = []
    for group_index, row in enumerate(groups):
        members = set(torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        group_contacts = tuple(
            edge for edge in contacts if edge.first in members and edge.second in members
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
