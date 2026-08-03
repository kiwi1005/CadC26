"""Pure dense pair features for geometry-aware collective dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hcfp.case import FloorplanCase


Tensor = torch.Tensor
RELATION_ORDER = ("left", "right", "above", "below")
PAIR_FEATURES = (
    "net_weight",
    "dx",
    "dy",
    "gap_left",
    "gap_right",
    "gap_above",
    "gap_below",
    "overlap_x",
    "overlap_y",
    "same_group",
    "same_mib",
    "topology_left",
    "topology_right",
    "topology_above",
    "topology_below",
    "latch_left",
    "latch_right",
    "latch_above",
    "latch_below",
)


@dataclass(frozen=True)
class PairFeatureBatch:
    features: Tensor
    pair_mask: Tensor


def dynamic_pair_features(
    case: FloorplanCase,
    center: Tensor,
    dimensions: Tensor,
    *,
    topology_relation: Tensor | None = None,
    active_latch: Tensor | None = None,
) -> PairFeatureBatch:
    """Build pair features with positive clearance for a satisfied relation.

    ``gap_left[i, j]`` is ``left_j - right_i``; the other relation gaps use
    the same convention in :data:`RELATION_ORDER`.  A negative value therefore
    means the selected relation is geometrically violated.
    """

    device = case.area.device
    if isinstance(center, Tensor) and center.device != device:
        raise ValueError("center must be on the case device")
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    if center.ndim != 3 or center.shape[1:] != (case.n, 2):
        raise ValueError("center must have shape [K,N,2]")
    k = center.shape[0]
    if isinstance(dimensions, Tensor) and dimensions.device != device:
        raise ValueError("dimensions must be on the case device")
    dimensions = torch.as_tensor(dimensions, dtype=torch.float32, device=device)
    if dimensions.shape != (k, case.n, 2):
        raise ValueError("dimensions must have shape [K,N,2]")
    if not bool(torch.isfinite(center).all() and torch.isfinite(dimensions).all()):
        raise ValueError("center and dimensions must be finite")
    if bool((dimensions <= 0.0).any()):
        raise ValueError("dimensions must be positive")

    pair_mask = ~torch.eye(case.n, dtype=torch.bool, device=center.device)
    first = center[:, :, None, :]
    second = center[:, None, :, :]
    first_dim = dimensions[:, :, None, :]
    second_dim = dimensions[:, None, :, :]
    delta = second - first
    half_sum = 0.5 * (first_dim + second_dim)
    overlap_x = (half_sum[..., 0] - delta[..., 0].abs()).clamp_min(0.0)
    overlap_y = (half_sum[..., 1] - delta[..., 1].abs()).clamp_min(0.0)

    left_i = first[..., 0] - 0.5 * first_dim[..., 0]
    right_i = first[..., 0] + 0.5 * first_dim[..., 0]
    bottom_i = first[..., 1] - 0.5 * first_dim[..., 1]
    top_i = first[..., 1] + 0.5 * first_dim[..., 1]
    left_j = second[..., 0] - 0.5 * second_dim[..., 0]
    right_j = second[..., 0] + 0.5 * second_dim[..., 0]
    bottom_j = second[..., 1] - 0.5 * second_dim[..., 1]
    top_j = second[..., 1] + 0.5 * second_dim[..., 1]

    topology = _relation_one_hot(
        topology_relation,
        k,
        case.n,
        center.device,
        "topology_relation",
    )
    latch = _relation_one_hot(
        active_latch,
        k,
        case.n,
        center.device,
        "active_latch",
    )
    features = torch.cat(
        (
            case.b2b_weight.to(device=center.device, dtype=torch.float32).expand(k, -1, -1).unsqueeze(-1),
            delta,
            (left_j - right_i).unsqueeze(-1),
            (left_i - right_j).unsqueeze(-1),
            (bottom_i - top_j).unsqueeze(-1),
            (bottom_j - top_i).unsqueeze(-1),
            overlap_x.unsqueeze(-1),
            overlap_y.unsqueeze(-1),
            _same_membership(case.group_membership, center.device).expand(k, -1, -1).unsqueeze(-1),
            _same_membership(case.mib_membership, center.device).expand(k, -1, -1).unsqueeze(-1),
            topology,
            latch,
        ),
        dim=-1,
    )
    if features.shape != (k, case.n, case.n, len(PAIR_FEATURES)):
        raise RuntimeError("internal pair feature shape mismatch")
    features = torch.where(pair_mask.view(1, case.n, case.n, 1), features, torch.zeros_like(features))
    return PairFeatureBatch(features=features, pair_mask=pair_mask)


def _same_membership(membership: Tensor, device: torch.device) -> Tensor:
    if not membership.numel():
        return torch.zeros((membership.shape[1], membership.shape[1]), dtype=torch.float32, device=device)
    active = membership.to(device=device, dtype=torch.float32)
    same = active.transpose(0, 1) @ active
    same = same > 0.0
    same.fill_diagonal_(False)
    return same.to(dtype=torch.float32)


def _relation_one_hot(
    relation: Tensor | None,
    k: int,
    n: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if relation is None:
        return torch.zeros((k, n, n, 4), dtype=torch.float32, device=device)
    if isinstance(relation, Tensor) and relation.device != device:
        raise ValueError(f"{name} must be on the case device")
    value = torch.as_tensor(relation, device=device)
    if value.shape == (k, n, n):
        if torch.is_floating_point(value) or value.dtype == torch.bool:
            raise ValueError(f"{name} relation ids must use an integer dtype")
        value = value.to(dtype=torch.long)
        valid = (value == -1) | ((0 <= value) & (value < 4))
        if not bool(valid.all()):
            raise ValueError(f"{name} must contain -1 or relation ids 0..3")
        return torch.nn.functional.one_hot(value.clamp_min(0), num_classes=4).to(dtype=torch.float32) * (value >= 0).unsqueeze(-1)
    if value.shape == (k, n, n, 4):
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} one-hot tensor must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        binary = (value == 0.0) | (value == 1.0)
        if not bool(binary.all()) or bool((value.sum(dim=-1) > 1.0).any()):
            raise ValueError(f"{name} one-hot rows must contain at most one active relation")
        return value.to(dtype=torch.float32)
    raise ValueError(f"{name} must have shape [K,N,N] or [K,N,N,4]")
