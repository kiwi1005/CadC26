from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hcfp.btree import BStarTree
from hcfp.btree_forest import btree_forest_candidates
from hcfp.verify import area_bad_blocks, overlap_pairs, verify_feasible


def _case(
    boxes: torch.Tensor, *, fixed: tuple[int, ...] = (), preplaced: tuple[int, ...] = ()
) -> SimpleNamespace:
    n = int(boxes.shape[0])
    return SimpleNamespace(
        n=n,
        area=boxes[:, 2] * boxes[:, 3],
        target=boxes.clone(),
        fixed_mask=torch.tensor([index in fixed for index in range(n)]),
        preplaced_mask=torch.tensor([index in preplaced for index in range(n)]),
        normalized=False,
        group_membership=torch.zeros((0, n), dtype=torch.bool),
        mib_membership=torch.zeros((0, n), dtype=torch.bool),
    )


def test_forest_returns_bounded_exact_feasible_candidates() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 2.0, 2.0),
            (5.0, 0.0, 1.0, 1.0),
            (8.0, 0.0, 1.0, 1.0),
            (12.0, 0.0, 1.0, 1.0),
        )
    )
    case = _case(boxes)
    tree = BStarTree.from_edges(torch.tensor(((0, 1, 0), (1, 2, 1), (2, 3, 0))), 4)

    candidates = btree_forest_candidates(case, boxes, tree=tree, max_candidates=4)

    assert 0 < len(candidates) <= 4
    for candidate in candidates:
        assert verify_feasible(case, candidate)
        assert not overlap_pairs(candidate)
        assert not area_bad_blocks(case, candidate)


def test_forest_preserves_fixed_and_preplaced_anchors() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 2.0, 2.0),
            (5.0, 0.0, 1.0, 1.0),
            (8.0, 0.0, 1.0, 1.0),
            (12.0, 0.0, 1.0, 1.0),
        )
    )
    case = _case(boxes, fixed=(0,), preplaced=(2,))
    candidates = btree_forest_candidates(case, boxes, max_candidates=8)

    assert candidates
    for candidate in candidates:
        assert torch.equal(candidate[0], boxes[0])
        assert torch.equal(candidate[2], boxes[2])
        assert verify_feasible(case, candidate)


def test_forest_rejects_invalid_tree_size_and_zero_budget() -> None:
    boxes = torch.tensor(((0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0)))
    case = _case(boxes)
    assert btree_forest_candidates(case, boxes, max_candidates=0) == ()
    with pytest.raises(ValueError, match="tree.block_count"):
        btree_forest_candidates(
            case,
            boxes,
            tree=BStarTree.from_edges(
                torch.tensor(((0, 1, 0), (1, 2, 1))),
                3,
            ),
            max_candidates=1,
        )
