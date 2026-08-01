"""Official FloorSet v10 input adapter for HCFP.

The adapter keeps the contest-facing inputs intact enough for evaluator parity
while building normalized tensors and derived masks for the HCFP runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


BOUNDARY_LEFT = 1
BOUNDARY_RIGHT = 2
BOUNDARY_TOP = 4
BOUNDARY_BOTTOM = 8
BOUNDARY_ORDER = (BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP, BOUNDARY_BOTTOM)


@dataclass(frozen=True)
class FloorplanCase:
    n: int
    area: torch.Tensor
    b2b_weight: torch.Tensor
    p2b_edges: torch.Tensor
    pins: torch.Tensor
    constraints: torch.Tensor
    target: torch.Tensor
    block_mask: torch.Tensor
    fixed_mask: torch.Tensor
    preplaced_mask: torch.Tensor
    target_valid_mask: torch.Tensor
    cluster_id: torch.Tensor
    mib_id: torch.Tensor
    group_membership: torch.Tensor
    mib_membership: torch.Tensor
    cluster_group_ids: torch.Tensor
    mib_group_ids: torch.Tensor
    boundary_bits: torch.Tensor
    scale: float
    origin: torch.Tensor

    @property
    def normalized(self) -> bool:
        """Coordinates and areas use ``origin``/``scale`` normalization."""

        return True

    def __post_init__(self) -> None:
        n = self.n
        if n <= 0:
            raise ValueError("block_count must be positive")
        _require_shape("area", self.area, (n,))
        _require_shape("b2b_weight", self.b2b_weight, (n, n))
        _require_shape("pins", self.pins, (None, 2))
        _require_shape("p2b_edges", self.p2b_edges, (None, 3))
        _require_shape("constraints", self.constraints, (n, 5))
        _require_shape("target", self.target, (n, 4))
        _require_shape("block_mask", self.block_mask, (n,))
        _require_shape("fixed_mask", self.fixed_mask, (n,))
        _require_shape("preplaced_mask", self.preplaced_mask, (n,))
        _require_shape("target_valid_mask", self.target_valid_mask, (n,))
        _require_shape("cluster_id", self.cluster_id, (n,))
        _require_shape("mib_id", self.mib_id, (n,))
        _require_shape("group_membership", self.group_membership, (None, n))
        _require_shape("mib_membership", self.mib_membership, (None, n))
        _require_shape("boundary_bits", self.boundary_bits, (n, 4))
        _require_shape("origin", self.origin, (2,))

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "FloorplanCase":
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if dtype is not None and torch.is_floating_point(value):
                    values[field.name] = value.to(device=device, dtype=dtype)
                else:
                    values[field.name] = value.to(device=device)
            else:
                values[field.name] = value
        return FloorplanCase(**values)


def from_official(
    block_count: int,
    area_targets: Any,
    b2b_connectivity: Any,
    p2b_connectivity: Any,
    pins_pos: Any,
    constraints: Any,
    target_positions: Any | None = None,
    *,
    device: torch.device | str | None = None,
) -> FloorplanCase:
    """Build a validated normalized case from the official v10 solve inputs."""

    n = int(block_count)
    if n <= 0:
        raise ValueError("block_count must be positive")

    raw_area = _as_float_tensor(area_targets, "area_targets").flatten()
    if raw_area.numel() < n:
        raise ValueError(f"area_targets must contain at least {n} entries")
    raw_area = raw_area[:n]
    if not torch.isfinite(raw_area).all() or (raw_area <= 0).any():
        raise ValueError("area_targets must be finite and positive for all unpadded blocks")

    constraint_tensor = _as_long_tensor(constraints, "constraints")
    if constraint_tensor.numel() == 0:
        constraint_tensor = torch.zeros((n, 5), dtype=torch.long)
    if constraint_tensor.ndim != 2 or constraint_tensor.shape[0] < n or constraint_tensor.shape[1] != 5:
        raise ValueError("constraints must have shape [>=block_count, 5]")
    constraint_tensor = constraint_tensor[:n].clone()
    _validate_constraints(constraint_tensor)

    fixed_mask = constraint_tensor[:, 0] > 0
    preplaced_mask = constraint_tensor[:, 1] > 0
    hard_mask = fixed_mask | preplaced_mask

    raw_target, target_valid_mask = _prepare_target(target_positions, n)
    if hard_mask.any() and target_positions is None:
        raise ValueError("target_positions is required when fixed or preplaced blocks are present")
    if hard_mask.any() and not target_valid_mask[hard_mask].all():
        raise ValueError("target_positions must be valid for every fixed or preplaced block")

    raw_pins = _unpad_pin_rows(_as_float_tensor(pins_pos, "pins_pos"))
    p2b_edges = _prepare_p2b_edges(p2b_connectivity, n, raw_pins.shape[0])
    b2b_weight = _prepare_b2b_weight(b2b_connectivity, n)

    scale = float(torch.sqrt(raw_area.sum()).item())
    origin = _compute_origin(raw_pins, raw_target, preplaced_mask)

    area = raw_area / (scale * scale)
    pins = (raw_pins - origin) / scale if raw_pins.numel() else raw_pins.reshape(0, 2)
    target = _normalize_target(raw_target, target_valid_mask, origin, scale)

    cluster_id = constraint_tensor[:, 3].clone()
    mib_id = constraint_tensor[:, 2].clone()
    cluster_group_ids, group_membership = _membership(cluster_id)
    mib_group_ids, mib_membership = _membership(mib_id)
    boundary_bits = _boundary_bits(constraint_tensor[:, 4])

    case = FloorplanCase(
        n=n,
        area=area.to(dtype=torch.float32),
        b2b_weight=b2b_weight.to(dtype=torch.float32),
        p2b_edges=p2b_edges.to(dtype=torch.float32),
        pins=pins.to(dtype=torch.float32),
        constraints=constraint_tensor,
        target=target.to(dtype=torch.float32),
        block_mask=torch.ones(n, dtype=torch.bool),
        fixed_mask=fixed_mask,
        preplaced_mask=preplaced_mask,
        target_valid_mask=target_valid_mask,
        cluster_id=cluster_id,
        mib_id=mib_id,
        group_membership=group_membership,
        mib_membership=mib_membership,
        cluster_group_ids=cluster_group_ids,
        mib_group_ids=mib_group_ids,
        boundary_bits=boundary_bits,
        scale=scale,
        origin=origin.to(dtype=torch.float32),
    )
    return case.to(device=device) if device is not None else case


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc


def _as_long_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value, dtype=torch.long, device="cpu")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc


def _prepare_b2b_weight(connectivity: Any, n: int) -> torch.Tensor:
    tensor = _as_float_tensor(connectivity, "b2b_connectivity")
    if tensor.numel() == 0:
        return torch.zeros((n, n), dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise ValueError("b2b edge list rows must contain [block_i, block_j, weight]")

    dense = torch.zeros((n, n), dtype=torch.float32)
    for row in tensor:
        i_raw, j_raw, weight = row[:3]
        if _is_padded_edge(i_raw, j_raw, weight):
            continue
        i = _checked_index(i_raw, n, "b2b block_i")
        j = _checked_index(j_raw, n, "b2b block_j")
        if i == j:
            raise ValueError("b2b self-edges are not allowed")
        if not torch.isfinite(weight) or float(weight) < 0.0:
            raise ValueError("b2b weights must be finite and non-negative")
        dense[i, j] += float(weight)
        dense[j, i] += float(weight)
    dense.fill_diagonal_(0.0)
    return dense


def _prepare_p2b_edges(connectivity: Any, n: int, pin_count: int) -> torch.Tensor:
    tensor = _as_float_tensor(connectivity, "p2b_connectivity")
    if tensor.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise ValueError("p2b_connectivity must have rows [pin, block, weight]")

    rows: list[tuple[float, float, float]] = []
    for row in tensor:
        pin_raw, block_raw, weight = row[:3]
        if _is_padded_edge(pin_raw, block_raw, weight):
            continue
        pin = _checked_index(pin_raw, pin_count, "p2b pin")
        block = _checked_index(block_raw, n, "p2b block")
        if not torch.isfinite(weight) or float(weight) < 0.0:
            raise ValueError("p2b weights must be finite and non-negative")
        rows.append((float(pin), float(block), float(weight)))
    if not rows:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _prepare_target(target_positions: Any | None, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    if target_positions is None:
        return torch.zeros((n, 4), dtype=torch.float32), torch.zeros(n, dtype=torch.bool)
    target = _as_float_tensor(target_positions, "target_positions")
    if target.ndim != 2 or target.shape[0] < n or target.shape[1] != 4:
        raise ValueError("target_positions must have shape [>=block_count, 4]")
    target = target[:n].clone()
    finite = torch.isfinite(target).all(dim=1)
    padded = (target == -1).all(dim=1)
    valid = finite & ~padded
    if (target[valid, 2:] <= 0).any():
        raise ValueError("valid target_positions must have positive width and height")
    target[~valid] = 0.0
    return target, valid


def _unpad_pin_rows(pins_pos: torch.Tensor) -> torch.Tensor:
    if pins_pos.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if pins_pos.ndim != 2 or pins_pos.shape[1] != 2:
        raise ValueError("pins_pos must have shape [P, 2]")
    keep = []
    for row in pins_pos:
        if (row == -1).all():
            break
        if not torch.isfinite(row).all():
            raise ValueError("pins_pos rows must be finite or [-1, -1] padding")
        keep.append(row)
    if not keep:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.stack(keep).to(dtype=torch.float32)


def _validate_constraints(constraints: torch.Tensor) -> None:
    fixed = constraints[:, 0]
    preplaced = constraints[:, 1]
    if ((fixed != 0) & (fixed != 1)).any():
        raise ValueError("constraints[:,0] fixed flags must be 0 or 1")
    if ((preplaced != 0) & (preplaced != 1)).any():
        raise ValueError("constraints[:,1] preplaced flags must be 0 or 1")
    if (constraints[:, 2] < -1).any() or (constraints[:, 3] < -1).any():
        raise ValueError("mib and cluster IDs must be -1 or non-negative")
    boundary = constraints[:, 4]
    if ((boundary < 0) | (boundary > 15)).any():
        raise ValueError("boundary bitmasks must be in [0, 15]")


def _compute_origin(pins: torch.Tensor, target: torch.Tensor, preplaced_mask: torch.Tensor) -> torch.Tensor:
    points: list[torch.Tensor] = []
    if preplaced_mask.any():
        preplaced = target[preplaced_mask]
        centers = preplaced[:, :2] + 0.5 * preplaced[:, 2:4]
        points.append(centers)
    if pins.numel():
        points.append(pins)
    if not points:
        return torch.zeros(2, dtype=torch.float32)
    return torch.cat(points, dim=0).mean(dim=0).to(dtype=torch.float32)


def _normalize_target(
    target: torch.Tensor,
    valid: torch.Tensor,
    origin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    normalized = torch.zeros_like(target, dtype=torch.float32)
    if valid.any():
        normalized[valid, :2] = (target[valid, :2] - origin) / scale
        normalized[valid, 2:4] = target[valid, 2:4] / scale
    return normalized


def _membership(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The official v10 scorer enumerates group ids from 1..max(id); 0 and -1
    # are both non-membership sentinels for unpadded blocks.
    group_ids = torch.unique(ids[ids > 0], sorted=True)
    if group_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty((0, ids.numel()), dtype=torch.bool)
    membership = group_ids[:, None] == ids[None, :]
    return group_ids.to(dtype=torch.long), membership


def _boundary_bits(codes: torch.Tensor) -> torch.Tensor:
    return torch.stack([(codes & bit) != 0 for bit in BOUNDARY_ORDER], dim=1)


def _checked_index(value: torch.Tensor, size: int, name: str) -> int:
    if not torch.isfinite(value):
        raise ValueError(f"{name} index must be finite")
    number = float(value)
    index = int(number)
    if number != index or index < 0 or index >= size:
        raise ValueError(f"{name} index {number:g} outside [0, {size})")
    return index


def _is_padded_edge(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> bool:
    return float(a) < 0.0 and float(b) < 0.0 and float(weight) < 0.0


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name} must have rank {len(expected)}")
    for actual, wanted in zip(tensor.shape, expected):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")
