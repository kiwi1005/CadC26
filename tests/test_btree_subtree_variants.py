from __future__ import annotations

import torch

from hcfp.btree import BStarTree, subtree_move_variants


def _tree() -> BStarTree:
    return BStarTree.from_edges(
        torch.tensor(
            (
                (0, 1, 0),
                (0, 2, 1),
                (1, 3, 0),
                (1, 4, 1),
                (2, 5, 0),
                (5, 6, 1),
            )
        ),
        7,
    )


def test_subtree_transpose_recursively_swaps_one_subtree() -> None:
    variants = dict(subtree_move_variants(_tree(), limit=32))

    candidate = variants["subtree_transpose:1"]
    assert candidate.root == 0
    assert candidate.left[0] == 1
    assert candidate.right[0] == 2
    assert candidate.left[1] == 4
    assert candidate.right[1] == 3
    assert candidate.left[2] == 5
    assert candidate.right[5] == 6
    assert BStarTree.from_edges(candidate.edges(), candidate.block_count) == candidate


def test_subtree_reinsert_stays_connected_and_avoids_cycles() -> None:
    tree = BStarTree.from_edges(
        torch.tensor(((0, 1, 0), (0, 2, 1), (1, 3, 0), (1, 4, 1))),
        5,
    )

    variants = subtree_move_variants(tree, limit=32)
    reinserts = [
        (name, candidate)
        for name, candidate in variants
        if name.startswith("subtree_reinsert:")
    ]

    assert reinserts
    for _, candidate in reinserts:
        assert (
            BStarTree.from_edges(candidate.edges(), candidate.block_count) == candidate
        )
        assert candidate.edges().shape == (candidate.block_count - 1, 3)


def test_subtree_move_variants_are_deterministic_and_bounded() -> None:
    tree = _tree()
    first = subtree_move_variants(tree, limit=5)
    second = subtree_move_variants(tree, limit=5)

    assert first == second
    assert len(first) <= 5
    assert len({(candidate.left, candidate.right) for _, candidate in first}) == len(
        first
    )
