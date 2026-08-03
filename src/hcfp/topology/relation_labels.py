"""Set-valued pairwise topology labels and losses."""

from __future__ import annotations

import math
from typing import Literal

import torch


Tensor = torch.Tensor
REL_LEFT = 0
REL_RIGHT = 1
REL_ABOVE = 2
REL_BELOW = 3
REL_UP = REL_ABOVE
REL_DOWN = REL_BELOW
RELATION_COUNT = 4
INVERSE_RELATION = (REL_RIGHT, REL_LEFT, REL_BELOW, REL_ABOVE)


def relation_mask_from_rectangles(
    rectangles: Tensor,
    *,
    valid_mask: Tensor | None = None,
    tolerance: float = 1.0e-7,
) -> Tensor:
    """Return all valid ``L/R/U/D`` relations as a boolean ``[N,N,4]`` mask."""

    boxes = torch.as_tensor(rectangles)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("rectangles must have shape [N,4]")
    if not torch.is_floating_point(boxes):
        boxes = boxes.to(dtype=torch.float32)
    elif boxes.dtype not in (torch.float32, torch.float64):
        boxes = boxes.float()
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0.0).any()):
        raise ValueError("rectangles must be finite with positive dimensions")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    n = boxes.shape[0]
    active = _node_mask(valid_mask, n, boxes.device)
    left, bottom = boxes[:, 0], boxes[:, 1]
    right, top = left + boxes[:, 2], bottom + boxes[:, 3]
    gaps = torch.stack(
        (
            left[None, :] - right[:, None],
            left[:, None] - right[None, :],
            bottom[:, None] - top[None, :],
            bottom[None, :] - top[:, None],
        ),
        dim=-1,
    )
    pair_mask = active[:, None] & active[None, :]
    pair_mask.fill_diagonal_(False)
    return (gaps >= -tolerance) & pair_mask.unsqueeze(-1)


def partial_label_nll(
    logits: Tensor,
    allowed_relations: Tensor,
    *,
    pair_mask: Tensor | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Negative log probability assigned to any allowed relation for each pair."""

    scores = _relation_logits(logits)
    try:
        allowed = torch.broadcast_to(
            torch.as_tensor(allowed_relations, dtype=torch.bool, device=scores.device),
            scores.shape,
        )
    except RuntimeError as exc:
        raise ValueError("allowed_relations must broadcast to logits shape") from exc

    selected = _pair_selection(pair_mask, scores)
    missing = selected & ~allowed.any(dim=-1)
    if bool(missing.any()):
        raise ValueError("every selected pair must allow at least one relation")
    if bool(selected.any()) and not bool(torch.isfinite(scores[selected]).all()):
        raise ValueError("selected logits must be finite")

    safe_scores = torch.where(selected.unsqueeze(-1), scores, torch.zeros_like(scores))
    safe_allowed = allowed | ~selected.unsqueeze(-1)
    allowed_scores = safe_scores.masked_fill(~safe_allowed, -torch.inf)
    loss = torch.logsumexp(safe_scores, dim=-1) - torch.logsumexp(
        allowed_scores, dim=-1
    )
    return _reduce(loss, selected, reduction)


def antisymmetry_loss(
    logits: Tensor,
    *,
    pair_mask: Tensor | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Penalize disagreement between ``p(i,j,r)`` and ``p(j,i,inverse(r))``."""

    scores = _relation_logits(logits)
    selected = _pair_selection(pair_mask, scores)
    selected = selected & selected.transpose(-2, -1)
    if bool(selected.any()) and not bool(torch.isfinite(scores[selected]).all()):
        raise ValueError("selected logits must be finite")

    safe_scores = torch.where(selected.unsqueeze(-1), scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    inverse = torch.tensor(INVERSE_RELATION, dtype=torch.long, device=scores.device)
    reverse = probabilities.transpose(-3, -2).index_select(-1, inverse)
    loss = (probabilities - reverse).square().sum(dim=-1)
    return _reduce(loss, selected, reduction)


def _relation_logits(logits: Tensor) -> Tensor:
    scores = torch.as_tensor(logits)
    if scores.ndim < 3 or scores.shape[-1] != RELATION_COUNT:
        raise ValueError("logits must have shape [...,N,N,4]")
    if scores.shape[-2] != scores.shape[-3]:
        raise ValueError("relation logits must be square in the block dimensions")
    if not torch.is_floating_point(scores):
        raise ValueError("logits must be floating point")
    return scores


def _node_mask(mask: Tensor | None, n: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    active = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if active.shape != (n,):
        raise ValueError("valid_mask must have shape [N]")
    return active


def _pair_selection(pair_mask: Tensor | None, logits: Tensor) -> Tensor:
    n = logits.shape[-2]
    shape = logits.shape[:-1]
    if pair_mask is None:
        selected = torch.ones(shape, dtype=torch.bool, device=logits.device)
    else:
        mask = torch.as_tensor(pair_mask, dtype=torch.bool, device=logits.device)
        if mask.shape == logits.shape[:-2]:
            mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        try:
            selected = torch.broadcast_to(mask, shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                "pair_mask must be a node mask or broadcast to [...,N,N]"
            ) from exc
    diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
    return selected & ~diagonal


def _reduce(
    loss: Tensor,
    selected: Tensor,
    reduction: Literal["none", "mean", "sum"],
) -> Tensor:
    if reduction == "none":
        return torch.where(selected, loss, torch.zeros_like(loss))
    if reduction == "sum":
        return loss[selected].sum() if bool(selected.any()) else loss.sum() * 0.0
    if reduction == "mean":
        return loss[selected].mean() if bool(selected.any()) else loss.sum() * 0.0
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")
