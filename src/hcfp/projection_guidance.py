"""Dense learned guidance sidecar for disjunctive projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.topology import decode_sequence_pair


Tensor = torch.Tensor
BDP_LEFT = 0
BDP_RIGHT = 1
BDP_BELOW = 2
BDP_ABOVE = 3
NEUTRAL = -1
INVERSE_BDP = (BDP_RIGHT, BDP_LEFT, BDP_ABOVE, BDP_BELOW)
TOPOLOGY_TO_BDP = (BDP_LEFT, BDP_RIGHT, BDP_ABOVE, BDP_BELOW)
SIDE_TO_ANCHOR_BDP = {
    "right": BDP_LEFT,
    "left": BDP_RIGHT,
    "above": BDP_BELOW,
    "below": BDP_ABOVE,
}


@dataclass(frozen=True)
class ProjectionGuidance:
    preferred_direction: Tensor
    preferred_confidence: Tensor
    contact_direction: Tensor
    contact_confidence: Tensor
    boundary_axis_lock: Tensor

    def __post_init__(self) -> None:
        _validate_direction_tensor("preferred_direction", self.preferred_direction)
        _validate_direction_tensor("contact_direction", self.contact_direction)
        if self.preferred_direction.shape != self.contact_direction.shape:
            raise ValueError("direction tensors must have matching [K,N,N] shapes")
        shape = tuple(self.preferred_direction.shape)
        _validate_confidence_tensor(
            "preferred_confidence",
            self.preferred_confidence,
            shape,
        )
        _validate_confidence_tensor(
            "contact_confidence",
            self.contact_confidence,
            shape,
        )
        if self.boundary_axis_lock.shape != (shape[0], shape[1], 2):
            raise ValueError("boundary_axis_lock must have shape [K,N,2]")
        if self.boundary_axis_lock.dtype != torch.bool:
            raise ValueError("boundary_axis_lock must be bool")
        _validate_inverse_consistency(
            "preferred",
            self.preferred_direction,
            self.preferred_confidence,
        )
        _validate_inverse_consistency(
            "contact",
            self.contact_direction,
            self.contact_confidence,
        )


def build_population_guidance(
    case: Any,
    provenance: dict[str, object],
    *,
    residual_count: int,
    constraint_count: int,
    topology_count: int,
) -> ProjectionGuidance:
    """Build guidance rows in learned population order: residual, constraint, topology."""

    counts = (residual_count, constraint_count, topology_count)
    if any(count < 0 for count in counts):
        raise ValueError("guidance counts must be non-negative")
    n = int(getattr(case, "n"))
    k = sum(counts)
    preferred = torch.full((k, n, n), NEUTRAL, dtype=torch.long)
    preferred_conf = torch.zeros((k, n, n), dtype=torch.float32)
    contact = preferred.clone()
    contact_conf = preferred_conf.clone()
    boundary_lock = torch.zeros((k, n, 2), dtype=torch.bool)

    topology_records = _records(
        provenance,
        "topology_seed_orders",
        topology_count,
    )
    topology_rows = [
        _topology_direction(record, provenance, n) for record in topology_records
    ]
    topology_offset = residual_count + constraint_count
    for index, row in enumerate(topology_rows):
        _assign_direction(
            preferred[topology_offset + index],
            preferred_conf[topology_offset + index],
            row,
        )

    constraint_records = _records(
        provenance,
        "constraint_seed_records",
        constraint_count,
    )
    for index, record in enumerate(constraint_records):
        topology_index = _index(record, "topology_seed_index", topology_count)
        row = residual_count + index
        _assign_direction(
            preferred[row],
            preferred_conf[row],
            topology_rows[topology_index],
        )
        _apply_contact_moves(contact[row], contact_conf[row], record, n)
        _apply_boundary_locks(boundary_lock[row], case, record, n)

    return ProjectionGuidance(
        preferred_direction=preferred,
        preferred_confidence=preferred_conf,
        contact_direction=contact,
        contact_confidence=contact_conf,
        boundary_axis_lock=boundary_lock,
    )


def _validate_direction_tensor(name: str, value: Tensor) -> None:
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f"{name} must have shape [K,N,N]")
    if value.dtype != torch.long:
        raise ValueError(f"{name} must be long")
    valid = (value == NEUTRAL) | ((value >= BDP_LEFT) & (value <= BDP_ABOVE))
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain -1 or BDP ids 0..3")


def _validate_confidence_tensor(
    name: str,
    value: Tensor,
    shape: tuple[int, ...],
) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must match direction tensor shape")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must be in [0,1]")


def _validate_inverse_consistency(
    name: str,
    direction: Tensor,
    confidence: Tensor,
) -> None:
    inverse = direction.clone()
    for value, other in enumerate(INVERSE_BDP):
        inverse[direction == value] = other
    expected = inverse.transpose(1, 2)
    if not bool(torch.equal(direction, expected)):
        raise ValueError(f"{name} directions must be inverse-consistent")
    if not bool(torch.allclose(confidence, confidence.transpose(1, 2))):
        raise ValueError(f"{name} confidences must be symmetric")
    diagonal = torch.diagonal(direction, dim1=1, dim2=2)
    if not bool((diagonal == NEUTRAL).all()):
        raise ValueError(f"{name} diagonal directions must be neutral")


def _records(
    provenance: dict[str, object],
    key: str,
    expected: int,
) -> tuple[dict[str, object], ...]:
    raw = provenance.get(key, ())
    if not isinstance(raw, (tuple, list)):
        raise ValueError(f"{key} must be a sequence")
    if len(raw) != expected:
        raise ValueError(f"{key} count mismatch")
    records: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{key} entries must be mappings")
        records.append(item)
    return tuple(records)


def _topology_direction(
    record: dict[str, object],
    provenance: dict[str, object],
    n: int,
) -> Tensor:
    try:
        order_hash = str(record["topology_order_sha256"])
        catalog = provenance["topology_order_catalog"]
    except KeyError as exc:
        raise ValueError("topology provenance is missing order metadata") from exc
    if not isinstance(catalog, dict):
        raise ValueError("topology_order_catalog must be a mapping")
    entry = catalog.get(order_hash)
    if not isinstance(entry, dict):
        raise ValueError("topology order is missing from catalog")
    try:
        positive = torch.tensor(entry["positive_order"], dtype=torch.long)
        negative = torch.tensor(entry["negative_order"], dtype=torch.long)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("catalog topology orders are malformed") from exc
    relation = decode_sequence_pair(positive, negative, n=n).relation
    mapped = torch.full((n, n), NEUTRAL, dtype=torch.long)
    for relation_id, direction_id in enumerate(TOPOLOGY_TO_BDP):
        mapped[relation == relation_id] = direction_id
    return mapped


def _assign_direction(target: Tensor, confidence: Tensor, value: Tensor) -> None:
    target.copy_(value)
    confidence[value >= 0] = 1.0


def _apply_contact_moves(
    direction: Tensor,
    confidence: Tensor,
    record: dict[str, object],
    n: int,
) -> None:
    for move in _moves(record):
        try:
            anchor = int(move["anchor"])
            child = int(move["child"])
            side = str(move["side"])
            direction_id = SIDE_TO_ANCHOR_BDP[side]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("contact move is malformed") from exc
        if not 0 <= anchor < n or not 0 <= child < n or anchor == child:
            raise ValueError("contact move endpoint is outside [0,N)")
        _set_pair(direction, confidence, anchor, child, direction_id)


def _moves(record: dict[str, object]) -> tuple[dict[str, object], ...]:
    details = record.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("constraint record details must be a mapping")
    raw = details.get("moves")
    if raw is None:
        group = details.get("group", {})
        if not isinstance(group, dict):
            raise ValueError("constraint record group details must be a mapping")
        raw = group.get("moves", ())
    if not isinstance(raw, (tuple, list)):
        raise ValueError("constraint record moves must be a sequence")
    moves: list[dict[str, object]] = []
    for move in raw:
        if not isinstance(move, dict):
            raise ValueError("constraint record moves must contain mappings")
        moves.append(move)
    return tuple(moves)


def _set_pair(
    direction: Tensor,
    confidence: Tensor,
    first: int,
    second: int,
    direction_id: int,
) -> None:
    if int(direction[first, second]) not in (NEUTRAL, direction_id):
        raise ValueError("contact move conflicts with an earlier contact")
    direction[first, second] = direction_id
    direction[second, first] = INVERSE_BDP[direction_id]
    confidence[first, second] = 1.0
    confidence[second, first] = 1.0


def _apply_boundary_locks(
    locks: Tensor,
    case: Any,
    record: dict[str, object],
    n: int,
) -> None:
    kind = str(record.get("kind", ""))
    if kind not in {"boundary_frame", "combined"}:
        return
    bits = torch.as_tensor(
        getattr(case, "boundary_bits"),
        dtype=torch.bool,
        device="cpu",
    )
    if bits.shape != (n, 4):
        raise ValueError("case boundary_bits must have shape [N,4]")
    placed = _placed_blocks(record, n)
    active = bits.any(dim=1)
    if placed is not None:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[list(placed)] = True
        active &= mask
    locks[:, 0] = active & (bits[:, 0] | bits[:, 1])
    locks[:, 1] = active & (bits[:, 2] | bits[:, 3])


def _placed_blocks(record: dict[str, object], n: int) -> tuple[int, ...] | None:
    details = record.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("constraint record details must be a mapping")
    source = details.get("boundary", details)
    if not isinstance(source, dict):
        raise ValueError("constraint boundary details must be a mapping")
    if "placed" not in source:
        return None
    raw = source["placed"]
    if not isinstance(raw, (tuple, list)):
        raise ValueError("boundary placed blocks must be a sequence")
    try:
        placed = tuple(int(block) for block in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("boundary placed blocks must be integer ids") from exc
    if any(not 0 <= block < n for block in placed):
        raise ValueError("boundary placed block is outside [0,N)")
    return placed


def _index(record: dict[str, object], key: str, limit: int) -> int:
    try:
        index = int(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"constraint record {key} is malformed") from exc
    if not 0 <= index < limit:
        raise ValueError(f"constraint record {key} is outside topology rows")
    return index
