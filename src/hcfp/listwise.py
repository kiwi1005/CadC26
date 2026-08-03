"""Listwise ranking losses for candidate-cost training."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def listmle_loss(
    predicted_score: Tensor,
    target_rank: Tensor,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Return ListMLE loss for one ranked candidate-cost list.

    ``predicted_score`` is a cost, so lower is better. The Plackett-Luce
    likelihood is computed over ``utility = -predicted_score`` ordered by
    ascending ``target_rank``.
    """

    _validate_inputs(predicted_score, target_rank, weight)
    order = torch.argsort(target_rank)
    utility = -predicted_score[order]
    denominator = torch.logcumsumexp(utility.flip(0), dim=0).flip(0)
    position_loss = denominator[:-1] - utility[:-1]
    if position_loss.numel() == 0:
        return position_loss.sum()
    if weight is None:
        return position_loss.mean()

    ordered_weight = weight[order][:-1].to(dtype=position_loss.dtype)
    total_weight = ordered_weight.sum()
    if not bool(total_weight > 0.0):
        raise ValueError("listmle weight must have positive mass on nontrivial positions")
    return (position_loss * ordered_weight).sum() / total_weight


def _validate_inputs(
    predicted_score: Tensor,
    target_rank: Tensor,
    weight: Tensor | None,
) -> None:
    if predicted_score.ndim != 1:
        raise ValueError("predicted_score must be a 1-D tensor")
    if target_rank.ndim != 1:
        raise ValueError("target_rank must be a 1-D tensor")
    if predicted_score.shape != target_rank.shape:
        raise ValueError("predicted_score and target_rank must have equal shape")
    if predicted_score.numel() == 0:
        raise ValueError("listmle_loss requires a nonempty candidate list")
    if predicted_score.device != target_rank.device:
        raise ValueError("predicted_score and target_rank must be on the same device")
    if not bool(torch.isfinite(predicted_score).all()):
        raise ValueError("predicted_score must be finite")
    if target_rank.dtype is not torch.long:
        raise ValueError("target_rank must have torch.long dtype")
    expected = torch.arange(target_rank.numel(), dtype=torch.long, device=target_rank.device)
    if not torch.equal(torch.sort(target_rank).values, expected):
        raise ValueError("target_rank must be an exact permutation of 0..K-1")
    if weight is None:
        return
    if weight.ndim != 1:
        raise ValueError("weight must be a 1-D tensor")
    if weight.shape != predicted_score.shape:
        raise ValueError("weight must match predicted_score shape")
    if weight.device != predicted_score.device:
        raise ValueError("weight must be on the same device as predicted_score")
    if weight.dtype is torch.bool:
        raise ValueError("weight must be numeric, not bool")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must be finite")
    if bool((weight < 0.0).any()):
        raise ValueError("weight must be nonnegative")
    if not bool(weight.sum() > 0.0):
        raise ValueError("weight must have positive total mass")


__all__ = ["listmle_loss"]
