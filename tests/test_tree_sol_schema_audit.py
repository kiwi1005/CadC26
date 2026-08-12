from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_tree_sol_schema import _contour_pack, _overlap_pairs, _parse_parent_child_side  # noqa: E402
from hcfp.btree import BStarTree  # noqa: E402


def test_parent_child_side_schema_and_contour_pack() -> None:
    tree = torch.tensor(((0, 1, 0), (0, 2, 1)), dtype=torch.float32)
    parsed = _parse_parent_child_side(tree, 3)
    assert parsed is not None
    root, left, right = parsed
    boxes = _contour_pack(
        root,
        left,
        right,
        torch.tensor(((2.0, 2.0), (1.0, 1.0), (2.0, 1.0))),
    )
    assert torch.equal(boxes[:, :2], torch.tensor(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))))
    assert _overlap_pairs(boxes) == 0


def test_parent_child_side_schema_rejects_duplicate_branch() -> None:
    tree = torch.tensor(((0, 1, 0), (0, 2, 0)), dtype=torch.float32)
    assert _parse_parent_child_side(tree, 3) is None


def test_runtime_btree_matches_audit_decoder() -> None:
    edges = torch.tensor(((0, 1, 0), (0, 2, 1)), dtype=torch.float32)
    dims = torch.tensor(((2.0, 2.0), (1.0, 1.0), (2.0, 1.0)))
    tree = BStarTree.from_edges(edges, 3)
    assert torch.equal(tree.pack(dims), _contour_pack(tree.root, tree.left, tree.right, dims))
