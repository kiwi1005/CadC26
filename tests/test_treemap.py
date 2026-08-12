from __future__ import annotations

from types import SimpleNamespace

import torch

from hcfp.case import from_official
from hcfp.geometry import overlap_area_matrix
from hcfp.treemap import exact_treemap_candidates
from hcfp.verify import (
    boundary_missing,
    mib_violation,
    verify_feasible,
)


def _hypothesis(bounds=(-0.5, -0.5, 0.5, 0.55)):
    return SimpleNamespace(
        bounds=bounds,
        confidence=1.0,
        hypothesis_id="treemap-test",
    )


def test_exact_treemap_tiles_outline_and_preserves_block_area() -> None:
    case = from_official(
        4,
        [1.0, 2.0, 3.0, 4.0],
        [],
        [],
        [],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    reference = torch.tensor(
        [
            [-0.4, -0.4, 0.1, 0.1],
            [-0.2, -0.2, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.1, 0.1],
        ]
    )

    candidates, records = exact_treemap_candidates(
        case, reference, (_hypothesis(),), count=4
    )

    assert candidates.shape == (4, 4, 4)
    assert len(records) == 4
    assert torch.allclose(
        candidates[..., 2] * candidates[..., 3],
        case.area.expand(4, -1),
        atol=1.0e-5,
    )
    assert float(overlap_area_matrix(candidates).max()) < 1.0e-7


def test_exact_treemap_restores_preplaced_geometry() -> None:
    case = from_official(
        3,
        [4.0, 3.0, 3.0],
        [],
        [],
        [],
        [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[-1.0, -1.0, 2.0, 2.0], [-1.0] * 4, [-1.0] * 4],
    )
    reference = torch.tensor(
        [
            [-1.0, -1.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    candidates, records = exact_treemap_candidates(
        case,
        reference,
        (_hypothesis(bounds=(-1.0, -1.0, 1.0, 1.625)),),
        count=1,
    )

    assert torch.equal(
        candidates[0, case.preplaced_mask], case.target[case.preplaced_mask]
    )
    assert records[0]["obstacle_aware"] is True
    assert float(overlap_area_matrix(candidates).max()) < 1.0e-7


def test_exact_treemap_constructs_compatible_mib_shape_before_slicing() -> None:
    case = from_official(
        4,
        [1.0, 1.0, 1.0, 1.0],
        [],
        [],
        [],
        [
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    )
    reference = torch.tensor(
        [
            [-0.8, -0.8, 0.5, 0.5],
            [-0.2, -0.2, 0.4, 0.6],
            [0.3, 0.3, 0.6, 0.4],
            [0.6, 0.6, 0.5, 0.5],
        ]
    )

    candidates, records = exact_treemap_candidates(
        case,
        reference,
        (_hypothesis(bounds=(-1.0, -1.0, 1.0, 1.0)),),
        count=1,
    )

    assert records[0]["mib_constructed_groups"] == (0,)
    assert torch.equal(candidates[0, 0, 2:4], case.target[0, 2:4])
    assert torch.equal(candidates[0, 0, 2:4], candidates[0, 1, 2:4])
    assert torch.equal(candidates[0, 1, 2:4], candidates[0, 2, 2:4])
    assert mib_violation(case, candidates[0]) == 0
    assert verify_feasible(case, candidates[0])


def test_exact_treemap_keeps_requested_outer_edges_on_candidate_bbox() -> None:
    case = from_official(
        2,
        [1.0, 1.0],
        [],
        [],
        [],
        [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2]],
    )
    reference = torch.tensor(
        [[-0.4, -0.2, 0.3, 0.3], [0.2, -0.2, 0.3, 0.3]]
    )

    candidates, _ = exact_treemap_candidates(
        case, reference, (_hypothesis(),), count=1
    )

    assert not bool(boundary_missing(case, candidates[0]).any())
    assert verify_feasible(case, candidates[0])
