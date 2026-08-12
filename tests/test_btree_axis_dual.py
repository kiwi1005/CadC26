from __future__ import annotations

import pytest
import torch

from hcfp.btree import BStarTree


_GUTTER = 2.0e-6


def _assert_hard_feasible(boxes: torch.Tensor, dimensions: torch.Tensor) -> None:
    assert torch.all(boxes[:, 2:] > 0)
    dimensions = dimensions.to(dtype=boxes.dtype)
    assert torch.allclose(boxes[:, 2:], dimensions)
    assert torch.isclose(
        (boxes[:, 2] * boxes[:, 3]).sum(),
        (dimensions[:, 0] * dimensions[:, 1]).sum(),
    )
    for first in range(boxes.shape[0]):
        for second in range(first):
            x_overlap = min(
                boxes[first, 0] + boxes[first, 2], boxes[second, 0] + boxes[second, 2]
            ) - max(boxes[first, 0], boxes[second, 0])
            y_overlap = min(
                boxes[first, 1] + boxes[first, 3], boxes[second, 1] + boxes[second, 3]
            ) - max(boxes[first, 1], boxes[second, 1])
            assert not (x_overlap > 1.0e-9 and y_overlap > 1.0e-9)


def test_y_compaction_is_the_transposed_x_compaction_with_y_tree_relations() -> None:
    tree = BStarTree.from_edges(
        torch.tensor(((0, 1, 0), (0, 2, 1), (1, 3, 1))),
        4,
    )
    dimensions = torch.tensor(((2.0, 3.0), (4.0, 1.0), (1.0, 5.0), (3.0, 2.0)))
    mask = torch.zeros(4, dtype=torch.bool)
    targets = torch.zeros((4, 4))
    origin = (4.0, 7.0)
    horizontal_order = torch.tensor((3, 2, 1, 0))

    x_boxes = tree.pack_x_compacted(
        dimensions,
        horizontal_order,
        mask,
        targets,
        origin=origin,
        gutter=_GUTTER,
    )
    y_boxes = tree.pack_y_compacted(
        dimensions,
        horizontal_order,
        mask,
        targets,
        origin=origin,
        gutter=_GUTTER,
    )

    transposed_tree = BStarTree(tree.root, tree.right, tree.left)
    expected_transposed = transposed_tree.pack_x_compacted(
        dimensions[:, [1, 0]],
        horizontal_order,
        mask,
        targets[:, [1, 0, 3, 2]],
        origin=(origin[1], origin[0]),
        gutter=_GUTTER,
    )
    assert torch.allclose(y_boxes[:, [1, 0, 3, 2]], expected_transposed)
    assert not torch.allclose(x_boxes, y_boxes)
    assert x_boxes[1, 0] == pytest.approx(x_boxes[0, 0] + dimensions[0, 0] + _GUTTER)
    assert y_boxes[2, 1] == pytest.approx(y_boxes[0, 1] + dimensions[0, 1] + _GUTTER)
    assert y_boxes[1, 1] == pytest.approx(y_boxes[0, 1])
    _assert_hard_feasible(x_boxes, dimensions)
    _assert_hard_feasible(y_boxes, dimensions)


def test_y_compaction_preserves_preplaced_box_identity_and_geometry() -> None:
    tree = BStarTree.from_edges(torch.tensor(((0, 1, 0), (0, 2, 1))), 3)
    dimensions = torch.tensor(((2.0, 2.0), (1.0, 1.0), (1.0, 5.0)))
    mask = torch.tensor((False, False, True))
    targets = torch.zeros((3, 4))
    targets[2] = torch.tensor((20.0, 30.0, 1.0, 5.0))

    boxes = tree.pack_y_compacted(
        dimensions,
        torch.tensor((2, 0, 1)),
        mask,
        targets,
        gutter=_GUTTER,
    )

    assert torch.equal(boxes[2], targets[2])
    _assert_hard_feasible(boxes, dimensions)
