"""Masked soft permutations and deterministic dependency-free hard decoding."""

from __future__ import annotations

import torch
from torch import nn


Tensor = torch.Tensor


def sinkhorn(
    logits: Tensor,
    mask: Tensor | None = None,
    *,
    iterations: int = 20,
    temperature: float = 1.0,
) -> Tensor:
    """Log-domain Sinkhorn normalization with zeroed padding rows and columns."""

    scores = torch.as_tensor(logits)
    if scores.ndim < 2 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("logits must have shape [...,N,N]")
    if not torch.is_floating_point(scores):
        raise ValueError("logits must be floating point")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    if mask is None:
        support = torch.ones_like(scores, dtype=torch.bool)
    else:
        try:
            support = torch.broadcast_to(
                torch.as_tensor(mask, dtype=torch.bool, device=scores.device),
                scores.shape,
            ).clone()
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to logits shape") from exc
    active_rows = support.any(dim=-1)
    active_columns = support.any(dim=-2)
    if not torch.equal(active_rows.sum(dim=-1), active_columns.sum(dim=-1)):
        raise ValueError("mask must expose equally many active rows and columns")
    complete_support = active_rows.unsqueeze(-1) & active_columns.unsqueeze(-2)
    if not torch.equal(support, complete_support):
        raise ValueError(
            "mask must be the complete active-row by active-column assignment support"
        )
    if bool(support.any()) and not bool(torch.isfinite(scores[support]).all()):
        raise ValueError("active assignment logits must be finite")

    work = scores if scores.dtype == torch.float64 else scores.float()
    work = torch.where(support, work / temperature, -torch.inf)
    for _ in range(iterations):
        row_norm = torch.logsumexp(work, dim=-1, keepdim=True)
        row_norm = torch.where(
            active_rows.unsqueeze(-1), row_norm, torch.zeros_like(row_norm)
        )
        work = torch.where(support, work - row_norm, work)
        column_norm = torch.logsumexp(work, dim=-2, keepdim=True)
        column_norm = torch.where(
            active_columns.unsqueeze(-2), column_norm, torch.zeros_like(column_norm)
        )
        work = torch.where(support, work - column_norm, work)
    return torch.where(support, work.exp(), torch.zeros_like(work))


class DualPermutationHead(nn.Module):
    """Map block embeddings to positive and negative soft sequence assignments."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_blocks: int = 120,
        sinkhorn_iterations: int = 20,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or max_blocks <= 0:
            raise ValueError("hidden_dim and max_blocks must be positive")
        if sinkhorn_iterations <= 0 or temperature <= 0.0:
            raise ValueError("Sinkhorn settings must be positive")
        self.max_blocks = max_blocks
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        self.positive = nn.Linear(hidden_dim, max_blocks)
        self.negative = nn.Linear(hidden_dim, max_blocks)

    def forward(
        self,
        embedding: Tensor,
        block_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        nodes = torch.as_tensor(embedding)
        if nodes.ndim not in (2, 3):
            raise ValueError("embedding must have shape [N,H] or [B,N,H]")
        single = nodes.ndim == 2
        work = nodes.unsqueeze(0) if single else nodes
        batch, n, _ = work.shape
        if n > self.max_blocks:
            raise ValueError(f"block count {n} exceeds max_blocks={self.max_blocks}")

        if block_mask is None:
            active = torch.ones((batch, n), dtype=torch.bool, device=work.device)
        else:
            active = torch.as_tensor(block_mask, dtype=torch.bool, device=work.device)
            if single and active.shape == (n,):
                active = active.unsqueeze(0)
            if active.shape != (batch, n):
                raise ValueError("block_mask must match [N] or [B,N]")
        counts = active.sum(dim=-1)
        if bool((counts <= 0).any()):
            raise ValueError("each batch item must contain at least one active block")
        ranks = torch.arange(n, device=work.device)
        active_ranks = ranks.unsqueeze(0) < counts.unsqueeze(1)
        support = active.unsqueeze(-1) & active_ranks.unsqueeze(-2)

        positive = sinkhorn(
            self.positive(work)[..., :n],
            support,
            iterations=self.sinkhorn_iterations,
            temperature=self.temperature,
        )
        negative = sinkhorn(
            self.negative(work)[..., :n],
            support,
            iterations=self.sinkhorn_iterations,
            temperature=self.temperature,
        )
        return (positive[0], negative[0]) if single else (positive, negative)


def greedy_hard_assignment(scores: Tensor, block_mask: Tensor | None = None) -> Tensor:
    """Assign each active block a rank with deterministic global-edge greedy.

    This is deliberately not Hungarian: it is dependency-free and deterministic,
    but it can miss the maximum-total-score assignment.
    """

    matrix = torch.as_tensor(scores)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scores must have shape [N,N]")
    if not torch.is_floating_point(matrix):
        raise ValueError("scores must be floating point")
    n = matrix.shape[0]
    active = _block_mask(block_mask, n, matrix.device)
    rows = torch.nonzero(active, as_tuple=False).flatten().tolist()
    rank_count = len(rows)
    if rank_count and not bool(torch.isfinite(matrix[active, :rank_count]).all()):
        raise ValueError("active assignment scores must be finite")

    # ponytail: greedy is O(N^2 log N) and suboptimal; use an explicitly allowed
    # Hungarian implementation if measured assignment regret justifies it.
    edges = [
        (-float(matrix[row, rank].detach()), int(row), int(rank))
        for row in rows
        for rank in range(rank_count)
    ]
    edges.sort()
    assignment = torch.full((n,), -1, dtype=torch.long, device=matrix.device)
    used_rows: set[int] = set()
    used_ranks: set[int] = set()
    for _, row, rank in edges:
        if row not in used_rows and rank not in used_ranks:
            assignment[row] = rank
            used_rows.add(row)
            used_ranks.add(rank)
            if len(used_rows) == rank_count:
                break
    if len(used_rows) != rank_count:
        raise ValueError("greedy assignment could not cover every active block")
    return assignment


def hard_permutation(scores: Tensor, block_mask: Tensor | None = None) -> Tensor:
    """Return active block ids ordered by their deterministic greedy rank."""

    assignment = greedy_hard_assignment(scores, block_mask)
    active_rows = torch.nonzero(assignment >= 0, as_tuple=False).flatten().tolist()
    active_rows.sort(key=lambda row: (int(assignment[row]), row))
    return torch.tensor(active_rows, dtype=torch.long, device=assignment.device)


def _block_mask(mask: Tensor | None, n: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    active = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if active.shape != (n,):
        raise ValueError("block_mask must have shape [N]")
    return active
