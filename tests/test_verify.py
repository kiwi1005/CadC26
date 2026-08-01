from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp import verify as v  # noqa: E402


@dataclass(frozen=True)
class Case:
    area: torch.Tensor
    target: torch.Tensor | None = None
    fixed_mask: torch.Tensor | None = None
    preplaced_mask: torch.Tensor | None = None
    boundary_codes: torch.Tensor | None = None
    group_membership: torch.Tensor | None = None
    mib_membership: torch.Tensor | None = None
    b2b_weight: torch.Tensor | None = None
    p2b_edges: torch.Tensor | None = None
    pins: torch.Tensor | None = None


def test_overlap_epsilon_and_edge_touching() -> None:
    legal_touch = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    tiny = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0 - 1.0e-10, 0.0, 1.0, 1.0]])
    positive = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.999, 0.0, 1.0, 1.0]])
    thin_but_official_overlap = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0 - 2.0e-6, 1.0 - 2.0e-6, 1.0, 1.0]],
        dtype=torch.float64,
    )

    assert v.overlap_pairs(legal_touch) == ()
    assert v.overlap_pairs(tiny) == ()
    assert v.overlap_pairs(positive) == ((0, 1),)
    assert v.overlap_pairs(thin_but_official_overlap) == ((0, 1),)


def test_hard_area_fixed_and_preplaced_tolerances() -> None:
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 3.0], [6.0, 0.0, 1.0, 1.0]])
    case = Case(
        area=torch.tensor([4.0, 6.0, 1.0]),
        target=boxes.clone(),
        fixed_mask=torch.tensor([False, True, False]),
        preplaced_mask=torch.tensor([False, False, True]),
    )

    assert v.verify(case, boxes).feasible
    bad = boxes.clone()
    bad[0, 2] = 2.03
    bad[1, 3] = 3.001
    bad[2, 0] = 6.001

    result = v.verify(case, bad)
    assert result.area_bad == (0,)
    assert result.fixed_bad == (1,)
    assert result.preplaced_bad == (2,)
    assert not result.feasible


def test_boundary_bitmask_uses_solution_bbox_all_codes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 0.5],
        ]
    )

    assert v.boundary_bitmask(boxes).tolist() == [9, 10, 5, 6, 0]


def test_group_edge_connectivity_and_mib_round4() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.00004, 1.0],
            [7.0, 0.0, 1.00006, 1.0],
        ]
    )
    case = Case(
        area=torch.ones(4),
        group_membership=torch.tensor([[True, True, False, False], [True, False, True, False]]),
        mib_membership=torch.tensor([[True, False, True, False], [False, False, True, True]]),
    )

    assert v.connected_components_for_group(boxes, torch.tensor([True, True, False, False])) == 1
    assert v.grouping_violation(case, boxes) == 1
    assert v.mib_shape_keys(boxes)[2:] == ((1.0, 1.0), (1.0001, 1.0))
    assert v.mib_violation(case, boxes) == 1


def test_b2b_p2b_hpwl_bbox_and_soft_violation_normalization() -> None:
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0], [3.0, 2.0, 2.0, 2.0]])
    case = Case(
        area=torch.tensor([4.0, 4.0, 4.0]),
        b2b_weight=torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 0.5], [1.0, 2.0, 2.0]]),
        pins=torch.tensor([[1.0, 1.0], [10.0, 5.0]]),
        boundary_codes=torch.tensor([v.BOUNDARY_LEFT, v.BOUNDARY_RIGHT, v.BOUNDARY_TOP]),
        group_membership=torch.tensor([[True, False, True]]),
        mib_membership=torch.tensor([[True, True, False]]),
    )

    assert v.bbox(boxes) == (0.0, 0.0, 5.0, 4.0)
    assert v.bbox_area(boxes) == 20.0
    assert v.b2b_hpwl(boxes, case.b2b_weight) == 12.0
    assert v.p2b_hpwl(boxes, case.p2b_edges, case.pins) == 16.0
    assert v.total_hpwl(case, boxes) == 28.0
    soft = v.soft_violation_normalized(case, boxes)
    assert soft.boundary == 0.0
    assert soft.grouping == 0.2
    assert soft.mib == 0.0
    assert soft.raw_grouping == 1
    assert soft.maximum == 5
    assert soft.total == 0.2
