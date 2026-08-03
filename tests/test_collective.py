from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hcfp.case import FloorplanCase, from_official
from hcfp.collective import PAIR_FEATURES, dynamic_pair_features


def _case() -> FloorplanCase:
    constraints = torch.tensor(
        [
            [0, 0, 7, 3, 0],
            [0, 0, 7, 3, 0],
            [0, 0, 0, 4, 0],
        ],
        dtype=torch.long,
    )
    return from_official(
        3,
        [4.0, 9.0, 16.0],
        [[0, 1, 2.5], [1, 2, 1.5]],
        [],
        [],
        constraints,
    )


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.tensor(
        [
            [[1.0, 1.0], [4.0, 1.0], [2.0, 5.0]],
            [[2.0, 1.0], [4.0, 2.0], [2.0, 4.0]],
        ]
    )
    dimensions = torch.tensor(
        [
            [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        ]
    )
    return center, dimensions


def _idx(name: str) -> int:
    return PAIR_FEATURES.index(name)


def _permute_case(case: FloorplanCase, perm: torch.Tensor) -> FloorplanCase:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    p2b = case.p2b_edges.clone()
    if p2b.numel():
        p2b[:, 1] = inv[p2b[:, 1].long()].to(dtype=p2b.dtype)
    return replace(
        case,
        area=case.area[perm],
        b2b_weight=case.b2b_weight[perm][:, perm],
        p2b_edges=p2b,
        constraints=case.constraints[perm],
        target=case.target[perm],
        block_mask=case.block_mask[perm],
        fixed_mask=case.fixed_mask[perm],
        preplaced_mask=case.preplaced_mask[perm],
        target_valid_mask=case.target_valid_mask[perm],
        cluster_id=case.cluster_id[perm],
        mib_id=case.mib_id[perm],
        group_membership=case.group_membership[:, perm],
        mib_membership=case.mib_membership[:, perm],
        boundary_bits=case.boundary_bits[perm],
    )


def test_pair_feature_contract_and_exact_geometry_channels() -> None:
    assert len(PAIR_FEATURES) == 19
    case = _case()
    center, dimensions = _geometry()
    topology = torch.full((2, case.n, case.n), -1, dtype=torch.long)
    topology[:, 0, 1] = 0
    topology[:, 1, 0] = 1
    latch = torch.zeros((2, case.n, case.n, 4), dtype=torch.float32)
    latch[:, 0, 2, 2] = 1.0

    batch = dynamic_pair_features(
        case,
        center,
        dimensions,
        topology_relation=topology,
        active_latch=latch,
    )

    assert batch.features.shape == (2, case.n, case.n, 19)
    assert batch.pair_mask.shape == (case.n, case.n)
    assert batch.features.device == center.device
    assert batch.pair_mask.diagonal().logical_not().all()
    assert torch.equal(batch.features[:, torch.arange(case.n), torch.arange(case.n)], torch.zeros((2, case.n, 19)))
    assert batch.features[0, 0, 1, _idx("net_weight")].item() == pytest.approx(2.5)
    assert batch.features[0, 0, 1, _idx("dx")].item() == pytest.approx(3.0)
    assert batch.features[0, 0, 1, _idx("dy")].item() == pytest.approx(0.0)
    assert batch.features[0, 0, 1, _idx("gap_left")].item() == pytest.approx(0.5)
    assert batch.features[0, 0, 1, _idx("gap_right")].item() == pytest.approx(-5.5)
    assert batch.features[0, 0, 2, _idx("gap_above")].item() == pytest.approx(-6.5)
    assert batch.features[0, 0, 2, _idx("gap_below")].item() == pytest.approx(1.5)
    assert torch.equal(
        batch.features[..., _idx("gap_left")],
        batch.features[..., _idx("gap_right")].transpose(1, 2),
    )
    assert torch.equal(
        batch.features[..., _idx("gap_above")],
        batch.features[..., _idx("gap_below")].transpose(1, 2),
    )
    assert batch.features[0, 0, 1, _idx("overlap_x")].item() == pytest.approx(0.0)
    assert batch.features[0, 0, 1, _idx("overlap_y")].item() == pytest.approx(2.0)
    assert batch.features[0, 0, 1, _idx("same_group")].item() == pytest.approx(1.0)
    assert batch.features[0, 0, 1, _idx("same_mib")].item() == pytest.approx(1.0)
    assert batch.features[0, 0, 1, _idx("topology_left")].item() == pytest.approx(1.0)
    assert batch.features[0, 0, 2, _idx("latch_above")].item() == pytest.approx(1.0)


def test_pair_features_are_translation_equivariant_and_recomputed() -> None:
    case = _case()
    center, dimensions = _geometry()
    first = dynamic_pair_features(case, center, dimensions).features
    moved = dynamic_pair_features(case, center + torch.tensor([13.0, -7.0]), dimensions).features
    assert torch.equal(first, moved)

    changed = center.clone()
    changed[:, 1, 0] += 1.0
    recomputed = dynamic_pair_features(case, changed, dimensions).features
    assert recomputed[0, 0, 1, _idx("dx")].item() == pytest.approx(first[0, 0, 1, _idx("dx")].item() + 1.0)
    assert not torch.equal(first, recomputed)


def test_pair_features_are_block_permutation_equivariant() -> None:
    case = _case()
    center, dimensions = _geometry()
    topology = torch.full((2, case.n, case.n), -1, dtype=torch.long)
    topology[:, 0, 1] = 0
    topology[:, 1, 0] = 1
    latch = torch.full((2, case.n, case.n), -1, dtype=torch.long)
    latch[:, 2, 0] = 3
    perm = torch.tensor([2, 0, 1])

    base = dynamic_pair_features(case, center, dimensions, topology_relation=topology, active_latch=latch)
    permuted = dynamic_pair_features(
        _permute_case(case, perm),
        center[:, perm],
        dimensions[:, perm],
        topology_relation=topology[:, perm][:, :, perm],
        active_latch=latch[:, perm][:, :, perm],
    )

    assert torch.equal(permuted.features, base.features[:, perm][:, :, perm])
    assert torch.equal(permuted.pair_mask, base.pair_mask[perm][:, perm])


def test_pair_feature_validation_rejects_bad_shapes_and_relations() -> None:
    case = _case()
    center, dimensions = _geometry()

    with pytest.raises(ValueError, match="center"):
        dynamic_pair_features(case, center[:, :2], dimensions)
    with pytest.raises(ValueError, match="dimensions"):
        dynamic_pair_features(case, center, dimensions[..., :1])
    bad = center.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        dynamic_pair_features(case, bad, dimensions)
    bad_dimensions = dimensions.clone()
    bad_dimensions[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        dynamic_pair_features(case, center, bad_dimensions)
    bad_relation = torch.zeros((2, case.n, case.n), dtype=torch.long)
    bad_relation[0, 0, 1] = 4
    with pytest.raises(ValueError, match="relation ids"):
        dynamic_pair_features(case, center, dimensions, topology_relation=bad_relation)
    with pytest.raises(ValueError, match="integer dtype"):
        dynamic_pair_features(case, center, dimensions, topology_relation=bad_relation.float())
    with pytest.raises(ValueError, match="active_latch"):
        dynamic_pair_features(case, center, dimensions, active_latch=torch.zeros((2, case.n, case.n, 3)))
    ambiguous_latch = torch.zeros((2, case.n, case.n, 4))
    ambiguous_latch[0, 0, 1, :2] = 1.0
    with pytest.raises(ValueError, match="at most one"):
        dynamic_pair_features(case, center, dimensions, active_latch=ambiguous_latch)
