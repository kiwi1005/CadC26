"""Bounded deterministic decoder for Contact repair actions."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from hcfp.constraints.contact_tree import extract_contacts
from hcfp.contact_synthesis import synthesize_contact_obligations
from hcfp.repair.actions import action_sha256
from hcfp.repair.decoders.base import DecodeFailure, DecodeResult
from hcfp.repair.schema import ExpertKind, RepairAction
from hcfp.verify import grouping_violation, verify_feasible


_RELATION = {"left": "LEFT", "right": "RIGHT", "above": "TOP", "below": "BOTTOM"}


def enumerate_contact_actions(
    case: Any,
    placement: Any,
    *,
    max_actions: int = 32,
) -> tuple[RepairAction, ...]:
    if max_actions <= 0:
        return ()
    synthesis = synthesize_contact_obligations(case, placement)
    actions = [
        RepairAction(
            expert=ExpertKind.CONTACT,
            obligation_id=f"contact-group:{obligation.group_index}",
            target_ids=(obligation.moving_member,),
            anchor_ids=(obligation.anchor_member,),
            relation=_RELATION[obligation.side],
            patch_budget=len(obligation.moving_component) + 1,
            score=-(
                obligation.move_distance
                + obligation.bbox_expansion
                - 1.0e-6 * obligation.net_incident
            ),
        )
        for obligation in synthesis.obligations + synthesis.candidate_edges
    ]
    return rank_contact_actions(actions)[:max_actions]


def rank_contact_actions(actions: Iterable[RepairAction]) -> tuple[RepairAction, ...]:
    unique: dict[str, RepairAction] = {}
    for action in actions:
        digest = action_sha256(action)
        incumbent = unique.get(digest)
        if incumbent is None or action.score > incumbent.score:
            unique[digest] = action
    return tuple(sorted(unique.values(), key=lambda action: (-action.score, action_sha256(action))))


def decode_contact_action(
    case: Any,
    placement: Any,
    action: RepairAction,
    *,
    verify_case: Any | None = None,
) -> DecodeResult:
    boxes = torch.as_tensor(placement, dtype=torch.float64, device="cpu").clone()
    before = grouping_violation(case, boxes)
    if (
        action.expert != ExpertKind.CONTACT
        or len(action.target_ids) != 1
        or len(action.anchor_ids) != 1
        or action.relation not in {"LEFT", "RIGHT", "TOP", "BOTTOM"}
    ):
        return DecodeResult(action, None, DecodeFailure.INVALID_ACTION, before, None)
    target, anchor = action.target_ids[0], action.anchor_ids[0]
    group = _common_group(case, target, anchor)
    if group is None or target == anchor:
        return DecodeResult(action, None, DecodeFailure.INVALID_ACTION, before, None)
    moving = _component(case, boxes, group, target)
    preplaced = torch.as_tensor(case.preplaced_mask, dtype=torch.bool, device="cpu")
    if bool(preplaced[list(moving)].any()):
        return DecodeResult(action, None, DecodeFailure.IMMOBILE_TARGET, before, None)
    if len(moving) + 1 > action.patch_budget:
        return DecodeResult(action, None, DecodeFailure.PATCH_BUDGET, before, None)

    destination = _target_xy(boxes[anchor], boxes[target], action.relation)
    candidate = boxes.clone()
    candidate[list(moving), :2] += destination - boxes[target, :2]
    if not verify_feasible(verify_case or case, candidate):
        return DecodeResult(action, None, DecodeFailure.HARD_INFEASIBLE, before, None, moving)
    after = grouping_violation(case, candidate)
    if after >= before:
        return DecodeResult(action, None, DecodeFailure.NO_DEBT_REDUCTION, before, after, moving)
    return DecodeResult(action, candidate, None, before, after, moving)


def _common_group(case: Any, first: int, second: int) -> torch.Tensor | None:
    membership = torch.as_tensor(case.group_membership, dtype=torch.bool, device="cpu")
    rows = torch.nonzero(membership[:, first] & membership[:, second], as_tuple=False).reshape(-1)
    return membership[int(rows[0])] if rows.numel() else None


def _component(case: Any, boxes: torch.Tensor, group: torch.Tensor, start: int) -> tuple[int, ...]:
    members = set(torch.nonzero(group, as_tuple=False).reshape(-1).tolist())
    adjacency = {member: set() for member in members}
    for contact in extract_contacts(boxes, tolerance=0.0):
        if contact.first in members and contact.second in members:
            adjacency[contact.first].add(contact.second)
            adjacency[contact.second].add(contact.first)
    found = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        new = adjacency[current] - found
        found.update(new)
        stack.extend(sorted(new, reverse=True))
    return tuple(sorted(found))


def _target_xy(anchor: torch.Tensor, target: torch.Tensor, relation: str) -> torch.Tensor:
    ax, ay, aw, ah = (float(value) for value in anchor)
    _, _, tw, th = (float(value) for value in target)
    y = min(max(float(target[1]), ay - th + 1.0e-7), ay + ah - 1.0e-7)
    x = min(max(float(target[0]), ax - tw + 1.0e-7), ax + aw - 1.0e-7)
    if relation == "RIGHT":
        return target.new_tensor((ax + aw, y))
    if relation == "LEFT":
        return target.new_tensor((ax - tw, y))
    if relation == "TOP":
        return target.new_tensor((x, ay + ah))
    return target.new_tensor((x, ay - th))
