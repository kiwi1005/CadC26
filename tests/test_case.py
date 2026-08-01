from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.case import from_official  # noqa: E402


def test_from_official_unpads_and_aggregates_duplicate_b2b_edges() -> None:
    case = from_official(
        block_count=3,
        area_targets=[4.0, 9.0, 16.0, -1.0],
        b2b_connectivity=[
            [0, 1, 1.5],
            [1, 0, 2.0],
            [0, 1, 0.5],
            [1, 2, 3.0],
            [-1, -1, -1],
        ],
        p2b_connectivity=[
            [0, 0, 2.0],
            [1, 2, 4.0],
            [-1, -1, -1],
        ],
        pins_pos=[
            [10.0, 0.0],
            [14.0, 2.0],
            [-1.0, -1.0],
        ],
        constraints=[
            [1, 0, 7, -1, 1 | 4],
            [0, 1, 7, 2, 2],
            [0, 0, -1, 2, 8],
            [-1, -1, -1, -1, -1],
        ],
        target_positions=[
            [1.0, 2.0, 2.0, 2.0],
            [8.0, 2.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    )

    scale = torch.sqrt(torch.tensor(29.0))
    expected_origin = torch.tensor([(9.5 + 10.0 + 14.0) / 3.0, (3.5 + 0.0 + 2.0) / 3.0])

    assert case.n == 3
    assert torch.all(case.block_mask)
    assert torch.allclose(case.area, torch.tensor([4.0, 9.0, 16.0]) / 29.0)
    assert torch.allclose(
        case.b2b_weight,
        torch.tensor(
            [
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 3.0],
                [0.0, 3.0, 0.0],
            ]
        ),
    )
    assert torch.equal(case.p2b_edges, torch.tensor([[0.0, 0.0, 2.0], [1.0, 2.0, 4.0]]))
    assert torch.allclose(case.origin, expected_origin)
    assert case.scale == pytest.approx(float(scale))
    assert torch.allclose(case.pins[0], (torch.tensor([10.0, 0.0]) - expected_origin) / scale)
    assert torch.allclose(case.target[1, :2], (torch.tensor([8.0, 2.0]) - expected_origin) / scale)
    assert torch.allclose(case.target[1, 2:], torch.tensor([3.0, 3.0]) / scale)

    assert torch.equal(case.fixed_mask, torch.tensor([True, False, False]))
    assert torch.equal(case.preplaced_mask, torch.tensor([False, True, False]))
    assert torch.equal(case.target_valid_mask, torch.tensor([True, True, False]))
    assert torch.equal(case.mib_group_ids, torch.tensor([7]))
    assert torch.equal(case.mib_membership, torch.tensor([[True, True, False]]))
    assert torch.equal(case.cluster_group_ids, torch.tensor([2]))
    assert torch.equal(case.group_membership, torch.tensor([[False, True, True]]))
    assert torch.equal(
        case.boundary_bits,
        torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, False, True],
            ]
        ),
    )


def test_zero_is_no_group_and_reverse_b2b_edges_are_aggregated() -> None:
    case = from_official(
        block_count=2,
        area_targets=[1.0, 1.0],
        b2b_connectivity=[[0, 1, 5.0], [1, 0, 7.0]],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 15]],
    )

    assert torch.equal(case.b2b_weight, torch.tensor([[0.0, 12.0], [12.0, 0.0]]))
    assert case.mib_group_ids.numel() == 0
    assert case.cluster_group_ids.numel() == 0
    assert case.mib_membership.shape == (0, 2)
    assert torch.equal(case.boundary_bits[1], torch.tensor([True, True, True, True]))
    assert torch.equal(case.origin, torch.zeros(2))


def test_missing_target_positions_for_hard_blocks_fails() -> None:
    with pytest.raises(ValueError, match="target_positions is required"):
        from_official(
            block_count=1,
            area_targets=[4.0],
            b2b_connectivity=[],
            p2b_connectivity=[],
            pins_pos=[],
            constraints=[[0, 1, -1, -1, 0]],
        )


def test_invalid_constraint_values_fail() -> None:
    with pytest.raises(ValueError, match="boundary bitmasks"):
        from_official(
            block_count=1,
            area_targets=[4.0],
            b2b_connectivity=[],
            p2b_connectivity=[],
            pins_pos=[],
            constraints=[[0, 0, -1, -1, 16]],
        )


def test_device_transfer_keeps_integer_and_bool_tensors_stable() -> None:
    case = from_official(
        block_count=1,
        area_targets=[4.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, -1, -1, 0]],
    )

    moved = case.to(device="cpu", dtype=torch.float64)

    assert moved.area.dtype == torch.float64
    assert moved.pins.dtype == torch.float64
    assert moved.constraints.dtype == torch.long
    assert moved.block_mask.dtype == torch.bool
    assert moved.n == case.n
