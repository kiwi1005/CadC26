from __future__ import annotations

import math

import pytest
import torch

from hcfp.topology import (
    INVERSE_RELATION,
    REL_ABOVE,
    REL_BELOW,
    REL_LEFT,
    REL_RIGHT,
    DualPermutationHead,
    adapt_preplaced_topology,
    antisymmetry_loss,
    check_preplaced_compatibility,
    copy_preplaced_targets,
    decode_sequence_pair,
    greedy_hard_assignment,
    hard_permutation,
    longest_path_coordinates,
    pack_sequence_pair,
    partial_label_nll,
    relation_mask_from_rectangles,
    sinkhorn,
)


def test_set_valued_relation_mask_and_partial_label_nll() -> None:
    rectangles = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 2.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )
    allowed = relation_mask_from_rectangles(rectangles)

    assert allowed.shape == (3, 3, 4)
    assert not allowed.diagonal(dim1=0, dim2=1).any()
    assert allowed[0, 1].tolist() == [True, False, False, True]
    assert allowed[1, 0].tolist() == [False, True, True, False]

    logits = torch.zeros((3, 3, 4), requires_grad=True)
    selected = torch.zeros((3, 3), dtype=torch.bool)
    selected[0, 1] = True
    loss = partial_label_nll(logits, allowed, pair_mask=selected)
    assert float(loss.detach()) == pytest.approx(math.log(2.0))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_relation_losses_honor_node_padding_and_antisymmetry() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((4, 4, 4), generator=generator)
    inverse = torch.tensor(INVERSE_RELATION)
    for first in range(4):
        for second in range(first + 1, 4):
            logits[second, first, inverse] = logits[first, second]

    node_mask = torch.tensor((True, True, True, False))
    assert float(antisymmetry_loss(logits, pair_mask=node_mask)) < 1.0e-12
    logits[1, 0, REL_RIGHT] += 2.0
    assert float(antisymmetry_loss(logits, pair_mask=node_mask)) > 0.0


def test_relation_losses_ignore_nonfinite_padding() -> None:
    logits = torch.zeros((3, 3, 4))
    logits[2] = torch.nan
    logits[:, 2] = torch.nan
    allowed = torch.zeros((3, 3, 4), dtype=torch.bool)
    allowed[0, 1, REL_LEFT] = True
    allowed[1, 0, REL_RIGHT] = True
    node_mask = torch.tensor((True, True, False))

    assert torch.isfinite(partial_label_nll(logits, allowed, pair_mask=node_mask))
    assert torch.isfinite(antisymmetry_loss(logits, pair_mask=node_mask))


def test_dual_permutation_head_is_padding_safe_and_deterministic() -> None:
    torch.manual_seed(3)
    head = DualPermutationHead(6, max_blocks=5, sinkhorn_iterations=40)
    embedding = torch.randn((2, 5, 6))
    block_mask = torch.tensor(
        (
            (True, True, False, True, False),
            (True, False, True, False, False),
        )
    )

    first = head(embedding, block_mask)
    second = head(embedding, block_mask)
    for lhs, rhs in zip(first, second):
        assert torch.equal(lhs, rhs)
        for batch, count in enumerate((3, 2)):
            active_rows = block_mask[batch]
            assert torch.allclose(
                lhs[batch, active_rows, :count].sum(dim=1),
                torch.ones(count),
                atol=2.0e-4,
            )
            assert torch.allclose(
                lhs[batch, :, :count].sum(dim=0), torch.ones(count), atol=2.0e-4
            )
            assert not lhs[batch, ~active_rows].any()
            assert not lhs[batch, :, count:].any()


def test_sinkhorn_rejects_incomplete_assignment_support() -> None:
    logits = torch.zeros((3, 3))
    hall_invalid = torch.tensor(
        (
            (True, False, False),
            (True, False, False),
            (False, True, True),
        )
    )

    with pytest.raises(ValueError, match="complete active-row"):
        sinkhorn(logits, hall_invalid)


def test_greedy_hard_assignment_is_stable_but_not_hungarian() -> None:
    scores = torch.tensor(((9.0, 8.0), (8.0, 0.0)))

    assignment = greedy_hard_assignment(scores)
    assert assignment.tolist() == [0, 1]
    assert hard_permutation(scores).tolist() == [0, 1]
    greedy_total = scores[torch.arange(2), assignment].sum()
    alternative_total = scores[0, 1] + scores[1, 0]
    assert greedy_total < alternative_total


def test_hard_assignment_supports_nonprefix_block_padding() -> None:
    scores = torch.tensor(
        (
            (0.1, 0.9, -9.0, -9.0),
            (8.0, 8.0, 8.0, 8.0),
            (0.8, 0.2, -9.0, -9.0),
            (8.0, 8.0, 8.0, 8.0),
        )
    )
    mask = torch.tensor((True, False, True, False))
    assignment = greedy_hard_assignment(scores, mask)

    assert assignment.tolist() == [1, -1, 0, -1]
    assert hard_permutation(scores, mask).tolist() == [2, 0]


def test_sequence_pair_decodes_inverse_relations_and_acyclic_edges() -> None:
    positive = torch.tensor((0, 1, 2, 3))
    negative = torch.tensor((1, 0, 3, 2))
    topology = decode_sequence_pair(positive, negative)

    assert int(topology.relation[0, 1]) == REL_ABOVE
    assert int(topology.relation[1, 0]) == REL_BELOW
    assert int(topology.relation[0, 2]) == REL_LEFT
    assert int(topology.relation[2, 0]) == REL_RIGHT
    assert _is_acyclic(4, topology.horizontal_edges)
    assert _is_acyclic(4, topology.vertical_edges)
    assert topology.horizontal_edges.shape[0] + topology.vertical_edges.shape[0] == 6


def test_longest_path_sequence_pack_is_compact_and_nonoverlapping() -> None:
    dimensions = torch.tensor(((2.0, 1.0), (1.0, 2.0), (1.0, 3.0)))
    rectangles = pack_sequence_pair(
        dimensions,
        positive=torch.tensor((0, 1, 2)),
        negative=torch.tensor((0, 2, 1)),
    )

    assert torch.equal(rectangles[:, 2:4], dimensions)
    assert rectangles[:, :2].tolist() == [[0.0, 0.0], [2.0, 3.0], [2.0, 0.0]]
    allowed = relation_mask_from_rectangles(rectangles)
    off_diagonal = ~torch.eye(3, dtype=torch.bool)
    assert allowed.any(dim=-1)[off_diagonal].all()


def test_longest_path_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        longest_path_coordinates(
            torch.ones(2),
            torch.tensor(((0, 1), (1, 0))),
        )


def test_low_confidence_preplaced_conflict_repairs_and_rechecks() -> None:
    positive = torch.tensor((1, 0, 2))
    negative = torch.tensor((1, 0, 2))
    targets = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    original_targets = targets.clone()
    mask = torch.tensor((True, True, False))
    confidence = torch.zeros((3, 3))
    confidence[0, 1] = confidence[1, 0] = 0.1

    report = check_preplaced_compatibility(positive, negative, targets, mask)
    assert not report.compatible
    assert [(item.first, item.second) for item in report.conflicts] == [(0, 1)]

    repaired = adapt_preplaced_topology(
        positive,
        negative,
        targets,
        mask,
        relation_confidence=confidence,
        repair_threshold=0.2,
    )
    assert check_preplaced_compatibility(*repaired, targets, mask).compatible
    assert torch.equal(targets, original_targets)


def test_preplaced_adaptation_rejects_unsafe_repairs() -> None:
    positive = torch.tensor((1, 0))
    negative = torch.tensor((1, 0))
    mask = torch.ones(2, dtype=torch.bool)
    targets = torch.tensor(((0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0)))
    confidence = torch.full((2, 2), 0.9)

    with pytest.raises(ValueError, match="high-confidence"):
        adapt_preplaced_topology(
            positive,
            negative,
            targets,
            mask,
            relation_confidence=confidence,
        )

    overlapping = torch.tensor(((0.0, 0.0, 2.0, 1.0), (1.0, 0.0, 2.0, 1.0)))
    with pytest.raises(ValueError, match="overlapping preplaced"):
        adapt_preplaced_topology(
            positive,
            negative,
            overlapping,
            mask,
            relation_confidence=torch.zeros((2, 2)),
        )


def test_preplaced_copy_is_exact_nonmutating_and_fail_closed() -> None:
    candidate = torch.tensor(
        (
            (9.0, 9.0, 1.0, 1.0),
            (2.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )
    original = candidate.clone()
    targets = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    mask = torch.tensor((True, False, False))

    copied = copy_preplaced_targets(candidate, targets, mask)
    assert torch.equal(copied[mask], targets[mask])
    assert torch.equal(candidate, original)

    colliding = candidate.clone()
    colliding[1, :2] = torch.tensor((0.5, 0.0))
    with pytest.raises(ValueError, match="repair rejected"):
        copy_preplaced_targets(colliding, targets, mask)

    with pytest.raises(ValueError, match="float32 or float64"):
        copy_preplaced_targets(candidate.to(torch.bfloat16), targets, mask)


def test_preplaced_adaptation_preserves_nonconflicting_movable_relations() -> None:
    positive = torch.tensor((1, 0, 2))
    negative = torch.tensor((1, 0, 2))
    targets = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    mask = torch.tensor((True, True, False))
    confidence = torch.zeros((3, 3))
    before = decode_sequence_pair(positive, negative).relation

    repaired = adapt_preplaced_topology(
        positive,
        negative,
        targets,
        mask,
        relation_confidence=confidence,
    )
    after = decode_sequence_pair(*repaired).relation

    assert int(after[0, 2]) == int(before[0, 2])
    assert int(after[1, 2]) == int(before[1, 2])


def _is_acyclic(n: int, edges: torch.Tensor) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    for source, target in edges.tolist():
        adjacency[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == n
