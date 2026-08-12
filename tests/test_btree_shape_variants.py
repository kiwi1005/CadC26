from __future__ import annotations

import pytest
import torch

from hcfp.btree import btree_dimension_variants


def _aspect_ratio(dimensions: torch.Tensor) -> torch.Tensor:
    width, height = dimensions[:, 0], dimensions[:, 1]
    return torch.maximum(width / height, height / width)


def test_dimension_variants_preserve_area_and_protected_shapes() -> None:
    dimensions = torch.tensor(
        ((128.0, 1.0), (1.0, 128.0), (96.0, 1.0), (80.0, 1.0)),
    )
    areas = dimensions.prod(dim=1).double()
    variants = btree_dimension_variants(
        dimensions,
        fixed_mask=torch.tensor((True, False, False, False)),
        preplaced_mask=torch.tensor((False, True, False, False)),
        mib_membership=torch.tensor(((False, False, True, False),)),
        areas=areas,
    )

    assert set(variants) == {"unlimited", "ar64", "ar32", "net_aware"}
    for candidate in variants.values():
        assert torch.allclose(candidate.prod(dim=1), areas, rtol=1e-10, atol=1e-10)
        assert torch.equal(candidate[:2], dimensions[:2])
        assert torch.equal(candidate[2], dimensions[2])
    assert _aspect_ratio(variants["ar64"])[3] <= 64.0 + 1e-10
    assert _aspect_ratio(variants["ar32"])[3] <= 32.0 + 1e-10


def test_net_aware_roles_use_weighted_degree() -> None:
    dimensions = torch.tensor(
        ((128.0, 1.0), (96.0, 1.0), (80.0, 1.0), (40.0, 1.0)),
    )
    variants = btree_dimension_variants(
        dimensions,
        weighted_degree=torch.tensor((10.0, 7.0, 3.0, 1.0)),
        high_degree_threshold=8.0,
        low_degree_threshold=2.0,
    )
    aware = variants["net_aware"]
    ratio = _aspect_ratio(aware)
    assert ratio[0] <= 32.0 + 1e-10
    assert ratio[1] <= 64.0 + 1e-10
    assert ratio[2] <= 64.0 + 1e-10
    assert ratio[3] == pytest.approx(40.0)
    assert torch.allclose(aware.prod(dim=1), dimensions.prod(dim=1).double())


def test_net_aware_protects_mib_members_even_when_high_degree() -> None:
    dimensions = torch.tensor(((128.0, 1.0), (96.0, 1.0)))
    variants = btree_dimension_variants(
        dimensions,
        mib_membership=torch.tensor(((True, False),)),
        weighted_degree=torch.tensor((100.0, 1.0)),
        high_degree_threshold=10.0,
        low_degree_threshold=0.0,
    )
    assert torch.equal(variants["ar32"][0], dimensions[0])
    assert torch.equal(variants["net_aware"][0], dimensions[0])
    assert _aspect_ratio(variants["net_aware"])[1] <= 64.0 + 1e-10
