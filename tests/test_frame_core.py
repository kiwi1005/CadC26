from __future__ import annotations

from types import SimpleNamespace

import torch

from hcfp.frame_core import (
    frame_core_candidates,
    frame_core_lns,
    identify_bbox_witnesses,
)
from hcfp.verify import grouping_violation, overlap_pairs, total_hpwl, verify_feasible


def _case(
    n: int,
    *,
    groups: tuple[tuple[int, ...], ...] = (),
    boundary: tuple[tuple[bool, bool, bool, bool], ...] | None = None,
    b2b: torch.Tensor | None = None,
    preplaced: tuple[int, ...] = (),
) -> SimpleNamespace:
    membership = torch.zeros((len(groups), n), dtype=torch.bool)
    for row, members in zip(membership, groups):
        row[list(members)] = True
    return SimpleNamespace(
        n=n,
        area=torch.ones(n),
        b2b_weight=torch.zeros((n, n)) if b2b is None else b2b,
        p2b_edges=torch.empty((0, 3)),
        pins=torch.empty((0, 2)),
        group_membership=membership,
        boundary_bits=torch.zeros((n, 4), dtype=torch.bool)
        if boundary is None
        else torch.tensor(boundary, dtype=torch.bool),
        preplaced_mask=torch.tensor([index in preplaced for index in range(n)]),
        fixed_mask=torch.zeros(n, dtype=torch.bool),
        normalized=False,
    )


def test_witnesses_prefer_boundary_members_already_on_frame() -> None:
    case = _case(
        3,
        boundary=(
            (False, False, False, True),
            (True, False, True, False),
            (False, True, False, False),
        ),
    )
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 2.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )

    witnesses = identify_bbox_witnesses(case, boxes)

    assert witnesses.left == 1
    assert witnesses.right == 2
    assert witnesses.top == 1
    assert witnesses.bottom == 0
    assert witnesses.indices == (0, 1, 2)


def test_frame_core_moves_group_component_without_expanding_frame() -> None:
    case = _case(3, groups=((0, 1),))
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (3.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )
    before_bounds = (0.0, 0.0, 5.0, 1.0)

    result = frame_core_lns(case, boxes, top_k=0, max_candidates=8)

    assert result.witnesses.bounds == before_bounds
    assert result.active == (1,)
    assert result.components == ((1,),)
    assert result.candidates
    candidate = result.candidates[0]
    assert grouping_violation(case, boxes) == 1
    assert grouping_violation(case, candidate.placement) == 0
    assert candidate.soft_after < candidate.soft_before
    assert verify_feasible(case, candidate.placement)
    assert not overlap_pairs(candidate.placement)
    assert torch.equal(candidate.placement[:, 2:4], boxes[:, 2:4])
    assert (candidate.placement[:, 0] >= 0.0).all()
    assert (candidate.placement[:, 0] + candidate.placement[:, 2] <= 5.0).all()
    assert torch.equal(candidate.placement[[0, 2]], boxes[[0, 2]])


def test_frame_core_protects_preplaced_witness_and_keeps_exact_geometry() -> None:
    case = _case(3, groups=((0, 1),), preplaced=(0,))
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (3.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )

    result = frame_core_lns(case, boxes, top_k=0, max_candidates=4)

    assert result.candidates
    for candidate in result.candidates:
        assert torch.equal(candidate.placement[0], boxes[0])
        assert torch.equal(candidate.placement[:, 2:4], boxes[:, 2:4])
        assert verify_feasible(case, candidate.placement)


def test_frame_core_active_set_adds_weighted_neighbors_but_not_witnesses() -> None:
    weights = torch.zeros((5, 5))
    weights[1, 3] = weights[3, 1] = 5.0
    case = _case(5, b2b=weights)
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
            (2.0, 1.0, 1.0, 1.0),
            (2.0, 3.0, 1.0, 1.0),
        )
    )

    result = frame_core_lns(case, boxes, top_k=1, max_candidates=4)

    assert result.witnesses.indices == (0, 2, 4)
    assert 1 in result.active
    assert 3 in result.active
    assert 0 not in result.active
    assert 2 not in result.active


def test_frame_core_accepts_hpwl_only_improvement_with_same_soft_count() -> None:
    weights = torch.zeros((3, 3))
    weights[0, 1] = weights[1, 0] = 4.0
    case = _case(3, b2b=weights)
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (3.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )
    before = total_hpwl(case, boxes)

    candidates = frame_core_candidates(case, boxes, top_k=1, max_candidates=8)

    assert candidates
    assert any(total_hpwl(case, candidate) < before for candidate in candidates)
    assert all(verify_feasible(case, candidate) for candidate in candidates)
