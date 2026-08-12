from __future__ import annotations

from types import SimpleNamespace

import torch

from hcfp.contact_patch import dense_contact_patch_candidates
from hcfp.verify import grouping_violation, verify_feasible


def test_dense_patch_reorders_without_whitespace() -> None:
    boxes = torch.tensor(
        [[float(index), 0.0, 1.0, 1.0] for index in range(8)],
        dtype=torch.float64,
    )
    group = torch.zeros((1, 8), dtype=torch.bool)
    group[0, [1, 6]] = True
    case = SimpleNamespace(
        n=8,
        normalized=False,
        area=torch.ones(8),
        fixed_mask=torch.zeros(8, dtype=torch.bool),
        preplaced_mask=torch.zeros(8, dtype=torch.bool),
        group_membership=group,
        mib_membership=torch.empty((0, 8), dtype=torch.bool),
        boundary_bits=torch.zeros((8, 4), dtype=torch.bool),
        b2b_weight=torch.zeros((8, 8)),
    )

    candidates = dense_contact_patch_candidates(
        case,
        boxes,
        patch_sizes=(8,),
        max_candidates=8,
    )

    assert candidates
    assert verify_feasible(case, candidates[0].placement)
    assert grouping_violation(case, candidates[0].placement) == 0
    assert torch.allclose(
        candidates[0].placement[:, 2:].prod(dim=1),
        torch.ones(8, dtype=torch.float64),
    )


def test_dense_patch_preserves_preplaced() -> None:
    boxes = torch.tensor(
        [[float(index), 0.0, 1.0, 1.0] for index in range(4)],
        dtype=torch.float64,
    )
    group = torch.tensor([[True, False, False, True]])
    case = SimpleNamespace(
        n=4,
        normalized=False,
        area=torch.ones(4),
        fixed_mask=torch.zeros(4, dtype=torch.bool),
        preplaced_mask=torch.tensor([True, False, False, False]),
        target=boxes.clone(),
        group_membership=group,
        mib_membership=torch.empty((0, 4), dtype=torch.bool),
        boundary_bits=torch.zeros((4, 4), dtype=torch.bool),
        b2b_weight=torch.zeros((4, 4)),
    )

    candidates = dense_contact_patch_candidates(case, boxes, patch_sizes=(4,))

    assert all(
        torch.equal(candidate.placement[0], boxes[0]) for candidate in candidates
    )
