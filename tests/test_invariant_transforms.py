from __future__ import annotations

import pytest
import torch

from hcfp.case import from_official
from hcfp.geometry import bbox_area_tensor, centers_from_xywh, overlap_area_matrix
from hcfp.invariant_transforms import (
    candidate_placements,
    mirror_candidates,
    reciprocal_affine_transform,
    weighted_median_translation,
)
from hcfp.verify import p2b_hpwl


def _case(*, p2b: bool = False, preplaced: bool = False):
    constraints = (
        [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        if preplaced
        else [[0, 0, 0, 0, 0]] * 3
    )
    return from_official(
        3,
        [4.0, 6.0, 9.0],
        [[0, 1, 2.0], [1, 2, 1.0]],
        [[0, 0, 2.0], [0, 1, 3.0]] if p2b else [],
        [[3.0, -2.0]] if p2b else [],
        constraints,
        [[0.0, 0.0, 2.0, 2.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]]
        if preplaced
        else None,
    )


def _placement() -> torch.Tensor:
    return torch.tensor(
        [
            [-0.8, -0.6, 0.4, 0.3],
            [0.1, -0.6, 0.2, 0.2],
            [0.5, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )


def _b2b_manhattan(case, boxes: torch.Tensor) -> torch.Tensor:
    centers = centers_from_xywh(boxes)
    pairwise = torch.abs(centers[:, None] - centers[None, :]).sum(dim=-1)
    return (torch.triu(case.b2b_weight, diagonal=1) * pairwise).sum()


def _assert_geometry_invariants(
    case, source: torch.Tensor, candidate: torch.Tensor
) -> None:
    assert torch.allclose(candidate[:, 2:].prod(dim=1), source[:, 2:].prod(dim=1))
    assert bool((torch.triu(overlap_area_matrix(candidate), diagonal=1) == 0.0).all())
    assert bbox_area_tensor(candidate) == pytest.approx(float(bbox_area_tensor(source)))
    assert _b2b_manhattan(case, candidate) == pytest.approx(
        float(_b2b_manhattan(case, source))
    )


def test_mirror_candidates_preserve_geometry_and_leave_boundary_filtering_to_caller() -> (
    None
):
    case = _case()
    source = _placement()
    candidates = mirror_candidates(source)

    assert len(candidates) == 4
    assert torch.equal(candidates[0], source)
    for candidate in candidates:
        _assert_geometry_invariants(case, source, candidate)


def test_reciprocal_affine_transform_preserves_area_nonoverlap_and_bbox_area() -> None:
    source = _placement()
    candidate = reciprocal_affine_transform(source, 2.0)

    assert torch.allclose(candidate[:, 2:].prod(dim=1), source[:, 2:].prod(dim=1))
    assert bool((torch.triu(overlap_area_matrix(candidate), diagonal=1) == 0.0).all())
    assert bbox_area_tensor(candidate) == pytest.approx(float(bbox_area_tensor(source)))
    assert not torch.equal(candidate[:, 2:], source[:, 2:])


def test_weighted_median_translation_does_not_worsen_p2b() -> None:
    case = _case(p2b=True)
    source = _placement()
    candidate = weighted_median_translation(case, source)

    before = p2b_hpwl(source, case.p2b_edges, case.pins)
    after = p2b_hpwl(candidate, case.p2b_edges, case.pins)
    assert after <= before + 1.0e-7
    _assert_geometry_invariants(case, source, candidate)


def test_translation_rejects_preplaced_and_candidate_api_is_minimal() -> None:
    source = _placement()
    with pytest.raises(ValueError, match="no preplaced"):
        weighted_median_translation(_case(p2b=True, preplaced=True), source)

    case = _case()
    candidates = candidate_placements(case, source, affine_scale=2.0)
    assert len(candidates) == 6
    assert all(candidate.shape == source.shape for candidate in candidates)
    assert all(
        torch.equal(candidate, expected)
        for candidate, expected in zip(candidates[:4], mirror_candidates(source))
    )
    assert torch.equal(candidates[4], source)
    assert torch.equal(candidates[5], reciprocal_affine_transform(source, 2.0))
