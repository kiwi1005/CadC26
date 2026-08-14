"""Losses for factorized Contact repair actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from hcfp.repair.model import CONTACT_RELATIONS, PATCH_BUDGETS, ContactActionOutput
from hcfp.repair.schema import ExpertKind, RepairAction


@dataclass(frozen=True)
class ContactActionLoss:
    total: torch.Tensor
    target: torch.Tensor
    anchor: torch.Tensor
    side: torch.Tensor
    patch_budget: torch.Tensor
    auxiliary: torch.Tensor


def contact_action_loss(
    output: ContactActionOutput,
    action: RepairAction,
    *,
    success: bool | None = None,
    debt_delta: float | None = None,
) -> ContactActionLoss:
    """Masked factorized NLL, with optional non-controlling value supervision."""

    target, anchor, side, patch = _indices(output, action)
    target_loss = F.cross_entropy(
        output.target_logits.unsqueeze(0),
        torch.tensor([target], dtype=torch.long, device=output.target_logits.device),
    )
    anchor_loss = F.cross_entropy(
        output.anchor_logits[target].unsqueeze(0),
        torch.tensor([anchor], dtype=torch.long, device=output.anchor_logits.device),
    )
    side_loss = F.cross_entropy(
        output.side_logits[target, anchor].unsqueeze(0),
        torch.tensor([side], dtype=torch.long, device=output.side_logits.device),
    )
    patch_loss = F.cross_entropy(
        output.patch_budget_logits[target, anchor].unsqueeze(0),
        torch.tensor(
            [patch], dtype=torch.long, device=output.patch_budget_logits.device
        ),
    )
    auxiliary = target_loss.new_zeros(())
    if success is not None:
        auxiliary = auxiliary + F.binary_cross_entropy_with_logits(
            output.success_logits[target, anchor],
            output.success_logits.new_tensor(float(success)),
        )
    if debt_delta is not None:
        auxiliary = auxiliary + F.mse_loss(
            output.debt_delta[target, anchor],
            output.debt_delta.new_tensor(float(debt_delta)),
        )
    return ContactActionLoss(
        total=target_loss + anchor_loss + side_loss + patch_loss + 0.1 * auxiliary,
        target=target_loss,
        anchor=anchor_loss,
        side=side_loss,
        patch_budget=patch_loss,
        auxiliary=auxiliary,
    )


def acceptable_contact_action_nll(
    output: ContactActionOutput, actions: tuple[RepairAction, ...]
) -> torch.Tensor:
    """Marginal NLL over every exact acceptable action for one state."""

    if not actions:
        raise ValueError("acceptable action set must not be empty")
    log_probabilities = torch.stack(
        tuple(-contact_action_loss(output, action).total for action in actions)
    )
    return -torch.logsumexp(log_probabilities, dim=0)


def _indices(
    output: ContactActionOutput, action: RepairAction
) -> tuple[int, int, int, int]:
    if (
        action.expert != ExpertKind.CONTACT
        or len(action.target_ids) != 1
        or len(action.anchor_ids) != 1
        or action.relation not in CONTACT_RELATIONS
        or action.patch_budget not in PATCH_BUDGETS
    ):
        raise ValueError("action is not a factorized Contact action")
    target, anchor = action.target_ids[0], action.anchor_ids[0]
    side = CONTACT_RELATIONS.index(action.relation)
    patch = PATCH_BUDGETS.index(action.patch_budget)
    if (
        target >= output.target_logits.numel()
        or anchor >= output.target_logits.numel()
        or not bool(output.masks.target[target])
        or not bool(output.masks.anchor[target, anchor])
        or not bool(output.masks.side[target, anchor, side])
        or not bool(output.masks.patch_budget[target, anchor, patch])
    ):
        raise ValueError("action is masked invalid for this repair state")
    return target, anchor, side, patch


__all__ = ["ContactActionLoss", "acceptable_contact_action_nll", "contact_action_loss"]
