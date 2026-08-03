"""Hard sequence-pair decoding into disjoint horizontal/vertical DAGs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hcfp.topology.relation_labels import REL_ABOVE, REL_BELOW, REL_LEFT, REL_RIGHT


Tensor = torch.Tensor
REL_NONE = -1


@dataclass(frozen=True)
class SequencePairTopology:
    relation: Tensor
    horizontal_edges: Tensor
    vertical_edges: Tensor
    active_mask: Tensor


def decode_sequence_pair(
    positive: Tensor,
    negative: Tensor,
    *,
    n: int | None = None,
) -> SequencePairTopology:
    """Decode two block-id permutations into pair relations and DAG edge lists."""

    plus = _permutation(positive, "positive")
    minus = _permutation(negative, "negative").to(device=plus.device)
    if plus.numel() != minus.numel() or not torch.equal(
        torch.sort(plus).values, torch.sort(minus).values
    ):
        raise ValueError(
            "positive and negative permutations must contain the same block ids"
        )
    inferred_n = int(plus.max()) + 1 if plus.numel() else 0
    block_count = inferred_n if n is None else int(n)
    if block_count < inferred_n or block_count <= 0:
        raise ValueError("n must be positive and include every permutation id")
    if n is None and plus.numel() != block_count:
        raise ValueError(
            "without n, permutations must contain exactly block ids 0..N-1"
        )

    active = torch.zeros(block_count, dtype=torch.bool, device=plus.device)
    active[plus] = True
    position_plus = torch.full((block_count,), -1, dtype=torch.long, device=plus.device)
    position_minus = position_plus.clone()
    order = torch.arange(plus.numel(), device=plus.device)
    position_plus[plus] = order
    position_minus[minus] = order

    before_plus = position_plus[:, None] < position_plus[None, :]
    before_minus = position_minus[:, None] < position_minus[None, :]
    after_plus = position_plus[:, None] > position_plus[None, :]
    after_minus = position_minus[:, None] > position_minus[None, :]
    valid_pair = active[:, None] & active[None, :]
    valid_pair.fill_diagonal_(False)

    relation = torch.full(
        (block_count, block_count), REL_NONE, dtype=torch.long, device=plus.device
    )
    relation[valid_pair & before_plus & before_minus] = REL_LEFT
    relation[valid_pair & after_plus & after_minus] = REL_RIGHT
    relation[valid_pair & before_plus & after_minus] = REL_ABOVE
    relation[valid_pair & after_plus & before_minus] = REL_BELOW
    if bool((relation[valid_pair] == REL_NONE).any()):
        raise ValueError("permutations did not define every active pair")

    first, second = torch.triu_indices(
        block_count, block_count, offset=1, device=plus.device
    )
    pair_active = active[first] & active[second]
    first, second = first[pair_active], second[pair_active]
    pair_relation = relation[first, second]

    horizontal = pair_relation <= REL_RIGHT
    horizontal_edges = torch.stack(
        (
            torch.where(
                pair_relation[horizontal] == REL_LEFT,
                first[horizontal],
                second[horizontal],
            ),
            torch.where(
                pair_relation[horizontal] == REL_LEFT,
                second[horizontal],
                first[horizontal],
            ),
        ),
        dim=1,
    )
    vertical = ~horizontal
    vertical_edges = torch.stack(
        (
            torch.where(
                pair_relation[vertical] == REL_ABOVE, second[vertical], first[vertical]
            ),
            torch.where(
                pair_relation[vertical] == REL_ABOVE, first[vertical], second[vertical]
            ),
        ),
        dim=1,
    )
    return SequencePairTopology(relation, horizontal_edges, vertical_edges, active)


def _permutation(value: Tensor, name: str) -> Tensor:
    raw = torch.as_tensor(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} permutation must have shape [M]")
    if torch.is_floating_point(raw):
        if not bool(torch.isfinite(raw).all()) or not bool((raw == raw.round()).all()):
            raise ValueError(f"{name} permutation must contain integer block ids")
    permutation = raw.to(dtype=torch.long)
    if (
        bool((permutation < 0).any())
        or torch.unique(permutation).numel() != permutation.numel()
    ):
        raise ValueError(
            f"{name} permutation must contain unique non-negative block ids"
        )
    return permutation
