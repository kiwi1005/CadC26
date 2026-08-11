from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.geometry import (
    centers_from_xywh,
    denormalize_xywh,
    exact_shape_projection,
    normalize_xywh,
    overlap_area_matrix,
)


def _case(device: str = "cpu"):
    return from_official(
        3,
        [4.0, 6.0, 9.0],
        [],
        [],
        [],
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 2.0, 3.0], [10.0, 4.0, 3.0, 3.0]],
        device=device,
    )


def test_shape_projection_preserves_area_and_hard_dimensions() -> None:
    case = _case()
    log_aspect = torch.tensor([[0.5, -0.8, 1.2], [-0.5, 0.8, -1.2]])
    wh = exact_shape_projection(case, log_aspect)

    assert torch.allclose(wh[:, 0, 0] * wh[:, 0, 1], case.area[0].expand(2))
    assert torch.equal(wh[:, 1], case.target[1, 2:4].expand(2, -1))
    assert torch.equal(wh[:, 2], case.target[2, 2:4].expand(2, -1))


def test_shape_projection_broadcasts_compatible_mib_shape() -> None:
    case = from_official(
        3,
        [4.0, 4.02, 9.0],
        [],
        [],
        [],
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
    )
    log_aspect = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.5, -0.5]])

    wh = exact_shape_projection(case, log_aspect)

    assert torch.equal(wh[:, 0], wh[:, 1])
    actual_area = wh[:, 0, 0] * wh[:, 0, 1]
    relative_error = torch.abs(actual_area[:, None] - case.area[:2]) / case.area[:2]
    assert bool((relative_error <= 1.0e-2).all())


def test_normalization_round_trip_and_overlap_matrix() -> None:
    case = _case()
    raw = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 3.0], [10.0, 4.0, 3.0, 3.0]])
    normalized = normalize_xywh(case, raw)
    restored = denormalize_xywh(case, normalized)

    assert torch.allclose(restored, raw, atol=1.0e-5)
    assert torch.allclose(centers_from_xywh(raw)[0], torch.tensor([1.0, 1.0]))
    overlap = overlap_area_matrix(raw)
    assert overlap[0, 1] == 1.0
    assert overlap[0, 2] == 0.0
