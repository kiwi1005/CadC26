from __future__ import annotations

from types import SimpleNamespace

import torch

from hcfp.boundary_skeleton import boundary_skeleton_candidates
from hcfp.verify import boundary_missing, verify_feasible


def _case(boundary_bits: torch.Tensor) -> SimpleNamespace:
    n = int(boundary_bits.shape[0])
    return SimpleNamespace(
        n=n,
        normalized=False,
        area=torch.ones(n),
        fixed_mask=torch.zeros(n, dtype=torch.bool),
        preplaced_mask=torch.zeros(n, dtype=torch.bool),
        group_membership=torch.empty((0, n), dtype=torch.bool),
        mib_membership=torch.empty((0, n), dtype=torch.bool),
        boundary_bits=boundary_bits,
        b2b_weight=torch.zeros((n, n)),
    )


def test_boundary_skeleton_reassigns_left_witness_inside_same_frame() -> None:
    boxes = torch.tensor(
        [[float(index), 0.0, 1.0, 1.0] for index in range(4)],
        dtype=torch.float64,
    )
    bits = torch.zeros((4, 4), dtype=torch.bool)
    bits[2, 0] = True
    case = _case(bits)

    candidates = boundary_skeleton_candidates(case, boxes, patch_sizes=(4,))

    assert candidates
    result = candidates[0].placement
    assert verify_feasible(case, result)
    assert int(torch.count_nonzero(boundary_missing(case, result))) == 0
    assert torch.allclose(
        result[:, 2:].prod(dim=1),
        torch.ones(4, dtype=torch.float64),
    )
    assert torch.isclose(result[:, 0].min(), boxes[:, 0].min())
    assert torch.isclose(
        (result[:, 0] + result[:, 2]).max(),
        (boxes[:, 0] + boxes[:, 2]).max(),
    )


def test_boundary_skeleton_can_make_corner_witness() -> None:
    boxes = torch.tensor(
        [[float(index), 0.0, 1.0, 1.0] for index in range(4)],
        dtype=torch.float64,
    )
    bits = torch.zeros((4, 4), dtype=torch.bool)
    bits[2, [0, 2]] = True
    case = _case(bits)

    candidates = boundary_skeleton_candidates(case, boxes, patch_sizes=(4,))

    assert candidates
    assert (
        int(torch.count_nonzero(boundary_missing(case, candidates[0].placement))) == 0
    )
