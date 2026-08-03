from __future__ import annotations

import math

import pytest
import torch

from hcfp.case import from_official
from hcfp.constraints.construction import (
    _component_overlaps_outside,
    connect_groups,
    construct_boundary_frame,
    construct_constraint_variants,
    construct_mib_shapes,
)
from hcfp.geometry import normalize_xywh
from hcfp.verify import boundary_missing, grouping_violation, mib_violation, verify_feasible


def test_group_construction_connects_disjoint_members_without_overlap() -> None:
    case = from_official(
        4,
        torch.ones(4),
        [],
        [],
        [],
        [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
    )
    raw = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0], [6.0, 0.0, 1.0, 1.0], [9.0, 0.0, 1.0, 1.0]]
    )
    boxes = normalize_xywh(case, raw)

    constructed, details = connect_groups(
        boxes,
        case.group_membership,
        preplaced_mask=case.preplaced_mask,
        b2b_weight=case.b2b_weight,
    )

    assert grouping_violation(case, boxes) == 2
    assert grouping_violation(case, constructed) == 0
    assert verify_feasible(case, constructed)
    assert details["move_count"] == 2
    assert details["unresolved_groups"] == ()


def test_group_construction_reports_two_anchored_components() -> None:
    targets = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
    )
    case = from_official(
        3,
        torch.ones(3),
        [],
        [],
        [],
        [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]],
        targets,
    )
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0], [8.0, 0.0, 1.0, 1.0]]
    )

    constructed, details = connect_groups(
        boxes,
        case.group_membership,
        preplaced_mask=case.preplaced_mask,
    )

    assert torch.equal(constructed[case.preplaced_mask], boxes[case.preplaced_mask])
    assert details["unresolved_groups"] == (0,)


def test_boundary_frame_satisfies_single_sides_and_corners_exactly() -> None:
    constraints = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 8],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 0, 10],
            [0, 0, 0, 0, 0],
        ]
    )
    case = from_official(7, torch.ones(7), [], [], [], constraints)
    raw = torch.tensor(
        [[float(index * 2), 0.0, 1.0, 1.0] for index in range(7)]
    )
    boxes = normalize_xywh(case, raw)

    constructed, details = construct_boundary_frame(
        boxes,
        case.boundary_bits,
        preplaced_mask=case.preplaced_mask,
    )

    assert not bool(boundary_missing(case, constructed).any())
    assert verify_feasible(case, constructed)
    assert details["reason"] == "ok"


def test_boundary_frame_fails_safe_when_required_anchor_is_preplaced() -> None:
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    case = from_official(
        2,
        torch.ones(2),
        [],
        [],
        [],
        [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]],
        target,
    )
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]])

    constructed, details = construct_boundary_frame(
        boxes,
        case.boundary_bits,
        preplaced_mask=case.preplaced_mask,
    )

    assert torch.equal(constructed, boxes)
    assert details["reason"] == "preplaced_boundary_anchor"


def test_mib_construction_shares_only_compatible_shapes() -> None:
    case = from_official(
        4,
        [2.0, 2.01, 2.0, 3.0],
        [],
        [],
        [],
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 2, 0, 0]],
    )
    raw = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0], [3.0, 0.0, 1.0, 2.01], [6.0, 0.0, 1.0, 2.0], [9.0, 0.0, 1.0, 3.0]]
    )
    boxes = normalize_xywh(case, raw)

    constructed, details = construct_mib_shapes(case, boxes)

    assert torch.equal(constructed[0, 2:4], constructed[1, 2:4])
    assert not torch.equal(constructed[2, 2:4], constructed[3, 2:4])
    assert mib_violation(case, constructed) == 1
    assert details["resolved_groups"] == (0,)
    assert details["incompatible_groups"] == (1,)


def test_mib_prediction_controls_compatible_group_shape() -> None:
    case = from_official(
        2,
        [2.0, 2.01],
        [],
        [],
        [],
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
    )
    boxes = normalize_xywh(
        case,
        torch.tensor(
            [[0.0, 0.0, 1.0, 2.0], [3.0, 0.0, 1.0, 2.01]]
        ),
    )

    square, square_details = construct_mib_shapes(
        case,
        boxes,
        mib_log_aspect=torch.tensor([0.0]),
    )
    wide, wide_details = construct_mib_shapes(
        case,
        boxes,
        mib_log_aspect=torch.tensor([math.log(4.0)]),
    )

    assert torch.equal(square[0, 2:4], square[1, 2:4])
    assert torch.equal(wide[0, 2:4], wide[1, 2:4])
    assert float(wide[0, 2] / wide[0, 3]) == pytest.approx(4.0)
    assert not torch.equal(square[:, 2:4], wide[:, 2:4])
    assert square_details["predicted_groups"] == (0,)
    assert wide_details["predicted_groups"] == (0,)


def test_variant_builder_retains_independent_constraint_sources() -> None:
    case = from_official(
        3,
        torch.ones(3),
        [],
        [],
        [],
        [[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
    )
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0], [6.0, 0.0, 1.0, 1.0]]
    )

    variants = construct_constraint_variants(case, boxes)

    assert {variant.kind for variant in variants} >= {"group_contacts", "boundary_frame"}
    assert all(variant.xywh.shape == boxes.shape for variant in variants)


@pytest.mark.parametrize(
    ("boxes", "component", "delta"),
    [
        pytest.param(
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]]),
            (0,),
            torch.tensor([1.0, 0.0]),
            id="edge-touch",
        ),
        pytest.param(
            torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]],
                dtype=torch.float64,
            ),
            (0,),
            torch.tensor(
                [
                    torch.nextafter(
                        torch.tensor(1.0, dtype=torch.float64),
                        torch.tensor(2.0, dtype=torch.float64),
                    ),
                    0.0,
                ]
            ),
            id="ulp-positive-overlap",
        ),
        pytest.param(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [2.5, 0.0, 1.0, 1.0],
                ]
            ),
            (0, 1),
            torch.tensor([1.0, 0.0]),
            id="nonchild-member-collision",
        ),
        pytest.param(
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]),
            (0, 1),
            torch.tensor([7.0, -3.0]),
            id="outside-empty",
        ),
    ],
)
def test_vectorized_component_overlap_matches_scalar_reference(
    boxes: torch.Tensor,
    component: tuple[int, ...],
    delta: torch.Tensor,
) -> None:
    expected = _scalar_component_overlaps_after_translation(boxes, component, delta)

    actual = _component_overlaps_outside(
        torch.as_tensor(boxes, dtype=torch.float64),
        component,
        torch.as_tensor(delta, dtype=torch.float64),
    )

    assert actual is expected


def _scalar_component_overlaps_after_translation(
    boxes: torch.Tensor,
    component: tuple[int, ...],
    delta: torch.Tensor,
) -> bool:
    candidate = torch.as_tensor(boxes, dtype=torch.float64).clone()
    candidate[list(component), :2] += torch.as_tensor(delta, dtype=torch.float64)
    inside = set(component)
    for first in component:
        ax, ay, aw, ah = (float(value) for value in candidate[first])
        for second in range(int(candidate.shape[0])):
            if second in inside:
                continue
            bx, by, bw, bh = (float(value) for value in candidate[second])
            if min(ax + aw, bx + bw) > max(ax, bx) and min(
                ay + ah, by + bh
            ) > max(ay, by):
                return True
    return False
