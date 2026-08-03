from __future__ import annotations

import math

import pytest
import torch

from hcfp.candidates import candidate_features
from hcfp.case import from_official


def _case():
    return from_official(
        3,
        torch.ones(3),
        [],
        [],
        [],
        torch.zeros((3, 5), dtype=torch.long),
    )


def _anchor() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_candidate_features_are_row_permutation_equivariant() -> None:
    case = _case()
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [4.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 3.0], [0.5, 1.0, 1.0, 1.0], [3.0, 0.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    permutation = torch.tensor([2, 0, 1])

    features = candidate_features(case, boxes, _anchor())
    permuted = candidate_features(case, boxes[permutation], _anchor())

    torch.testing.assert_close(permuted, features[permutation], rtol=0.0, atol=0.0)


def test_duplicate_geometries_get_identical_candidate_features() -> None:
    case = _case()
    duplicate = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [4.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    boxes = torch.stack((duplicate, duplicate.clone()))

    features = candidate_features(case, boxes, _anchor())

    torch.testing.assert_close(features[0], features[1], rtol=0.0, atol=0.0)


def test_candidate_feature_last_column_counts_positive_overlap_pairs() -> None:
    case = _case()
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 1.0, 1.0]],
            [[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    features = candidate_features(case, boxes, _anchor())

    expected = torch.tensor([math.log1p(1.0), math.log1p(3.0)], dtype=torch.float32)
    torch.testing.assert_close(features[:, 7], expected, rtol=0.0, atol=0.0)


def test_candidate_features_validate_shape_and_stay_finite() -> None:
    case = _case()
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )

    features = candidate_features(case, boxes, _anchor())

    assert features.shape == (1, 8)
    assert torch.isfinite(features).all()
    with pytest.raises(ValueError, match=r"\[C,N,4\]"):
        candidate_features(case, boxes[:, :, :3], _anchor())
