from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hcfp.region_assignment import obstacle_region_candidates
from hcfp.verify import bbox_area, verify_feasible


def test_region_assignment_splits_one_island_across_obstacle_regions() -> None:
    reference = torch.tensor(
        (
            (2.0, 0.0, 2.0, 2.0),
            (0.0, 3.0, 2.0, 2.0),
            (2.0, 3.0, 2.0, 2.0),
        )
    )
    case = SimpleNamespace(
        n=3,
        area=torch.tensor((4.0, 4.0, 4.0)),
        target=reference.clone(),
        fixed_mask=torch.zeros(3, dtype=torch.bool),
        preplaced_mask=torch.tensor((True, False, False)),
        group_membership=torch.empty((0, 3), dtype=torch.bool),
        mib_membership=torch.empty((0, 3), dtype=torch.bool),
        boundary_bits=torch.zeros((3, 4), dtype=torch.bool),
        b2b_weight=torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
        normalized=False,
    )
    outline = SimpleNamespace(
        bounds=(0.0, 0.0, 6.0, 2.0),
        confidence=1.0,
        hypothesis_id="split-test",
    )

    candidates, records = obstacle_region_candidates(
        case, reference, (outline,), count=2
    )

    assert candidates.shape[0] > 0
    assert any(record.split_members == (1, 2) for record in records)
    for candidate in candidates:
        assert verify_feasible(case, candidate)
        assert torch.equal(candidate[0], reference[0])
        assert bbox_area(candidate) == pytest.approx(12.0, rel=1.0e-5)


def test_region_assignment_moves_fixed_shape_without_resizing() -> None:
    reference = torch.tensor(
        (
            (2.0, 0.0, 2.0, 2.0),
            (0.0, 3.0, 1.0, 4.0),
            (1.0, 3.0, 2.0, 2.0),
        )
    )
    case = SimpleNamespace(
        n=3,
        area=torch.tensor((4.0, 4.0, 4.0)),
        target=reference.clone(),
        fixed_mask=torch.tensor((False, True, False)),
        preplaced_mask=torch.tensor((True, False, False)),
        group_membership=torch.empty((0, 3), dtype=torch.bool),
        mib_membership=torch.empty((0, 3), dtype=torch.bool),
        boundary_bits=torch.zeros((3, 4), dtype=torch.bool),
        b2b_weight=torch.zeros((3, 3)),
        normalized=False,
    )
    outline = SimpleNamespace(
        bounds=(0.0, 0.0, 8.0, 4.0),
        confidence=1.0,
        hypothesis_id="fixed-test",
    )

    candidates, _ = obstacle_region_candidates(case, reference, (outline,), count=2)

    assert candidates.shape[0] > 0
    assert all(
        torch.equal(candidate[1, 2:4], reference[1, 2:4]) for candidate in candidates
    )
    assert any(
        not torch.equal(candidate[1, :2], reference[1, :2]) for candidate in candidates
    )
