from __future__ import annotations

import dataclasses

import pytest
import torch

from hcfp.case import from_official
from hcfp.constraints.boundary_slots import construct_boundary_slots
from hcfp.constraints.latching import (
    LatchConfig,
    LatchState,
    orthogonal_overlap,
    side_gap,
    side_predicate,
    update_latch,
)
from hcfp.constraints.mib_shapes import resolve_mib_shapes


def test_hysteretic_latch_requires_consecutive_on_and_releases_deterministically() -> None:
    config = LatchConfig(epsilon_on=0.05, epsilon_off=0.20, consecutive_on=2)
    state = LatchState()

    first = update_latch(state, 0.04, config)
    second = update_latch(first, 0.03, config)
    held = update_latch(second, 0.10, config)
    released = update_latch(held, 0.20, config)

    assert state == LatchState()
    assert first == LatchState(active=False, consecutive=1)
    assert second == LatchState(active=True, consecutive=2)
    assert held == LatchState(active=True, consecutive=0)
    assert released == LatchState(active=False, consecutive=0)
    assert dataclasses.is_dataclass(released)


def test_latch_deadband_resets_inactive_consecutive_count() -> None:
    config = LatchConfig(epsilon_on=0.05, epsilon_off=0.20, consecutive_on=2)
    partial = update_latch(LatchState(), 0.04, config)
    reset = update_latch(partial, 0.10, config)
    still_inactive = update_latch(reset, 0.04, config)

    assert reset == LatchState(active=False, consecutive=0)
    assert still_inactive == LatchState(active=False, consecutive=1)


def test_latch_rejects_non_hysteretic_thresholds() -> None:
    with pytest.raises(ValueError, match="epsilon_on"):
        LatchConfig(epsilon_on=0.1, epsilon_off=0.1)


def test_side_gap_and_orthogonal_overlap_predicates() -> None:
    left = torch.tensor([0.0, 0.0, 2.0, 2.0])
    right = torch.tensor([2.05, 0.5, 1.0, 1.0])
    above_without_overlap = torch.tensor([3.5, 2.0, 1.0, 1.0])

    assert side_gap(left, right, "right") == pytest.approx(0.05)
    assert side_gap(right, left, "left") == pytest.approx(0.05)
    assert orthogonal_overlap(left, right, "right")
    assert not orthogonal_overlap(left, above_without_overlap, "top")
    assert side_predicate(left, right, "right", max_gap=0.051)
    assert not side_predicate(left, right, "right", max_gap=0.049)


def test_boundary_slots_follow_current_bit_semantics_and_keep_corner_membership() -> None:
    case = from_official(
        4,
        [1.0, 1.0, 1.0, 1.0],
        [],
        [],
        [],
        [
            [0, 0, 0, 0, 9],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 6],
            [0, 0, 0, 0, 2],
        ],
    )
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 1.0],
            [3.0, 3.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
        ]
    )

    constraints = construct_boundary_slots(case.boundary_bits, boxes)

    assert [(slot.block, slot.side, slot.ordinal) for slot in constraints.slots] == [
        (0, "left", 0),
        (1, "left", 1),
        (3, "right", 0),
        (2, "right", 1),
        (2, "top", 0),
        (0, "bottom", 0),
    ]
    assert {(item.block, item.block_side) for item in constraints.equalities} == {
        (0, "left"),
        (0, "bottom"),
        (1, "left"),
        (2, "right"),
        (2, "top"),
        (3, "right"),
    }
    assert [(item.before, item.after, item.side) for item in constraints.orders] == [
        (0, 1, "left"),
        (3, 2, "right"),
    ]


def test_boundary_slot_ties_break_by_secondary_coordinate_then_block_index() -> None:
    bits = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    boxes = torch.tensor(
        [
            [2.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )

    constraints = construct_boundary_slots(bits, boxes)

    assert [(slot.block, slot.ordinal) for slot in constraints.slots] == [(1, 0), (2, 1), (0, 2)]
    assert [(item.before, item.after) for item in constraints.orders] == [(1, 2), (2, 0)]


def test_mib_soft_group_gets_exact_shared_shape_within_one_percent_area_tolerance() -> None:
    result = resolve_mib_shapes(
        torch.tensor([2.0, 2.01, 4.0]),
        torch.tensor([[True, True, False]]),
        proposed_wh=torch.tensor([[1.0, 2.0], [1.1, 1.9], [2.0, 2.0]]),
    )

    assert torch.equal(result.shapes[0], result.shapes[1])
    assert torch.all(torch.abs(result.shapes[:2].prod(dim=1) - torch.tensor([2.0, 2.01])) / torch.tensor([2.0, 2.01]) <= 0.01)
    assert result.groups[0].compatible
    assert result.incompatible_groups == ()


def test_mib_hard_member_anchors_shape_and_reports_area_incompatibility() -> None:
    result = resolve_mib_shapes(
        torch.tensor([4.0, 9.0]),
        torch.tensor([[True, True]]),
        proposed_wh=torch.tensor([[2.5, 1.6], [1.5, 6.0]]),
        hard_mask=torch.tensor([True, False]),
        hard_wh=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
    )

    assert torch.equal(result.shapes[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(result.shapes[1], torch.tensor([1.5, 6.0]))
    assert not result.groups[0].compatible
    assert result.groups[0].shape is None
    assert result.groups[0].anchor == 0
    assert result.groups[0].reason == "hard shape violates member area tolerance"


def test_mib_conflicting_hard_shapes_are_reported_and_first_hard_shape_wins() -> None:
    result = resolve_mib_shapes(
        torch.tensor([4.0, 9.0]),
        torch.tensor([[True, True]]),
        hard_mask=torch.tensor([True, True]),
        hard_wh=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
    )

    assert torch.equal(result.shapes[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(result.shapes[1], torch.tensor([3.0, 3.0]))
    assert result.incompatible_groups[0].shape is None
    assert result.incompatible_groups[0].reason == "conflicting hard shapes"


def test_mib_soft_group_with_no_common_area_tolerance_is_incompatible() -> None:
    result = resolve_mib_shapes(
        torch.tensor([2.0, 3.0]),
        torch.tensor([[True, True]]),
    )

    assert not result.groups[0].compatible
    assert result.groups[0].shape is None
    assert torch.allclose(result.shapes, torch.tensor([[2.0**0.5, 2.0**0.5], [3.0**0.5, 3.0**0.5]]))
    assert result.groups[0].reason == "empty area-tolerance intersection"
