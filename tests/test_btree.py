from __future__ import annotations

import pytest
import torch

from hcfp.btree import (
    BStarTree,
    contact_aware_vertical_orders,
    decode_btree_logits,
    local_tree_variants,
)


def test_btree_parses_and_packs_without_overlap() -> None:
    tree = BStarTree.from_edges(
        torch.tensor(((0, 1, 0), (0, 2, 1))),
        3,
    )
    boxes = tree.pack(torch.tensor(((2.0, 2.0), (1.0, 1.0), (2.0, 1.0))))
    assert boxes[:, :2].tolist() == [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]


def test_btree_rejects_cycles_and_duplicate_branches() -> None:
    with pytest.raises(ValueError):
        BStarTree.from_edges(torch.tensor(((0, 1, 0), (0, 2, 0))), 3)
    with pytest.raises(ValueError):
        BStarTree.from_edges(torch.tensor(((0, 1, 0), (1, 0, 0))), 3)


def test_btree_preloads_preplaced_obstacles_into_contour() -> None:
    tree = BStarTree.from_edges(torch.tensor(((0, 1, 0), (1, 2, 0))), 3)
    boxes = tree.pack_with_preplaced(
        torch.tensor(((2.0, 2.0), (2.0, 2.0), (2.0, 2.0))),
        torch.tensor((False, True, False)),
        torch.tensor(((0.0, 0.0, 2.0, 2.0), (2.0, 3.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0))),
    )
    assert boxes[1].tolist() == [2.0, 3.0, 2.0, 2.0]
    assert boxes[0, 1] == 0.0
    assert boxes[2, 1] == 0.0


def test_btree_x_compaction_uses_runtime_vertical_order() -> None:
    tree = BStarTree.from_edges(torch.tensor(((0, 1, 0), (0, 2, 1))), 3)
    boxes = tree.pack_x_compacted(
        torch.tensor(((2.0, 2.0), (2.0, 3.0), (2.0, 1.0))),
        torch.tensor((2, 0, 1)),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros((3, 4)),
    )
    assert boxes[2, 1] == 0.0
    assert boxes[0, 1] > boxes[2, 1]
    assert boxes[1, 1] == 0.0


def test_contact_aware_orders_prioritize_boundary_and_cluster_groups() -> None:
    base = torch.tensor([0, 1, 2, 3])
    boundary = torch.zeros((4, 4), dtype=torch.bool)
    boundary[2, 3] = True
    boundary[0, 2] = True
    groups = torch.tensor([[False, True, False, True]])

    variants = dict(contact_aware_vertical_orders(base, boundary, groups))

    assert variants["boundary_band"].tolist() == [2, 1, 3, 0]
    assert variants["group_cluster"].tolist() == [0, 1, 3, 2]
    combined = variants.get("boundary_group", variants["boundary_band"])
    assert combined.tolist()[0] == 2


def test_local_tree_variants_stay_connected_and_reinsert_group_leaf() -> None:
    tree = BStarTree.from_edges(
        torch.tensor(((0, 1, 0), (0, 2, 1), (1, 3, 0))),
        4,
    )
    boundary = torch.zeros((4, 4), dtype=torch.bool)
    boundary[1, 0] = True
    groups = torch.tensor([[False, True, True, False]])

    variants = local_tree_variants(tree, boundary, groups, limit=8)

    assert variants
    assert any(name.startswith("sibling_flip") for name, _ in variants)
    assert any(name.startswith("group_reinsert") for name, _ in variants)
    for _, candidate in variants:
        assert candidate.block_count == 4
        assert candidate.pack(torch.ones((4, 2))).shape == (4, 4)


def test_logit_decoder_always_builds_a_valid_binary_tree() -> None:
    torch.manual_seed(9)
    tree = decode_btree_logits(torch.randn(7), torch.randn(7, 7, 2))

    assert tree.block_count == 7
    assert tree.edges().shape == (6, 3)
    assert BStarTree.from_edges(tree.edges(), 7) == tree
