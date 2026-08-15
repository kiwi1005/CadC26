"""Derive exact dynamic repair state from a placement."""

from __future__ import annotations

from typing import Any

import torch

from hcfp.case import BOUNDARY_BOTTOM, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP
from hcfp.constraints.contact_tree import extract_contacts
from hcfp.repair.schema import RepairState
from hcfp.verify import boundary_missing, edge_connected, mib_shape_keys


_SIDE_CODE = {
    "LEFT": BOUNDARY_LEFT,
    "RIGHT": BOUNDARY_RIGHT,
    "TOP": BOUNDARY_TOP,
    "BOTTOM": BOUNDARY_BOTTOM,
}


def build_repair_state(
    case,
    placement: Any,
    *,
    geometry_observed: Any | None = None,
    repair_target: Any | None = None,
    exact_contact_placement: Any | None = None,
    round_index: int = 0,
    corruption_kind: str | None = None,
    corruption_level: int = 0,
) -> RepairState:
    boxes = torch.as_tensor(placement, dtype=torch.float32, device="cpu").reshape(
        case.n, 4
    )
    contact_boxes = (
        boxes
        if exact_contact_placement is None
        else torch.as_tensor(
            exact_contact_placement, dtype=torch.float64, device="cpu"
        ).reshape(case.n, 4)
    )
    if not bool(torch.isfinite(contact_boxes).all()) or not bool(
        (contact_boxes[:, 2:4] > 0).all()
    ):
        raise ValueError(
            "exact Contact placement must be finite with positive dimensions"
        )
    observed = _mask(geometry_observed, case.n, default=True)
    target = _mask(repair_target, case.n, default=False)
    preplaced = case.preplaced_mask.detach().cpu().bool()
    fixed = case.fixed_mask.detach().cpu().bool()
    contacts = extract_contacts(
        contact_boxes,
        net_weight=case.b2b_weight,
        tolerance=0.0,
    )
    edges = torch.tensor(
        [
            (
                edge.first,
                edge.second,
                _SIDE_CODE[edge.first_side],
                _SIDE_CODE[edge.second_side],
            )
            for edge in contacts
        ],
        dtype=torch.long,
    ).reshape(-1, 4)
    return RepairState(
        case=case,
        placement=boxes,
        geometry_observed=observed,
        repair_target=target,
        position_mobility=~preplaced,
        shape_mobility=~(fixed | preplaced),
        contact_edges=edges,
        group_component_id=_group_components(case, contact_boxes),
        boundary_missing=boundary_missing(case, boxes).long(),
        mib_shape_class=_mib_shape_classes(case, boxes),
        round_index=round_index,
        corruption_kind=corruption_kind,
        corruption_level=corruption_level,
    )


def state_to_payload(state: RepairState) -> dict[str, Any]:
    from hcfp.data import case_to_payload

    return {
        "case": case_to_payload(state.case),
        "placement": state.placement.tolist(),
        "geometry_observed": state.geometry_observed.tolist(),
        "repair_target": state.repair_target.tolist(),
        "position_mobility": state.position_mobility.tolist(),
        "shape_mobility": state.shape_mobility.tolist(),
        "contact_edges": state.contact_edges.tolist(),
        "group_component_id": state.group_component_id.tolist(),
        "boundary_missing": state.boundary_missing.tolist(),
        "mib_shape_class": state.mib_shape_class.tolist(),
        "round_index": state.round_index,
        "corruption_kind": state.corruption_kind,
        "corruption_level": state.corruption_level,
    }


def state_from_payload(payload: dict[str, Any]) -> RepairState:
    from hcfp.data import case_from_payload

    return RepairState(
        case=case_from_payload(payload["case"]),
        placement=torch.as_tensor(payload["placement"], dtype=torch.float32),
        geometry_observed=torch.as_tensor(
            payload["geometry_observed"], dtype=torch.bool
        ),
        repair_target=torch.as_tensor(payload["repair_target"], dtype=torch.bool),
        position_mobility=torch.as_tensor(
            payload["position_mobility"], dtype=torch.bool
        ),
        shape_mobility=torch.as_tensor(payload["shape_mobility"], dtype=torch.bool),
        contact_edges=torch.as_tensor(
            payload["contact_edges"], dtype=torch.long
        ).reshape(-1, 4),
        group_component_id=torch.as_tensor(
            payload["group_component_id"], dtype=torch.long
        ),
        boundary_missing=torch.as_tensor(payload["boundary_missing"], dtype=torch.long),
        mib_shape_class=torch.as_tensor(payload["mib_shape_class"], dtype=torch.long),
        round_index=int(payload.get("round_index", 0)),
        corruption_kind=(
            str(payload["corruption_kind"])
            if payload.get("corruption_kind") is not None
            else None
        ),
        corruption_level=int(payload.get("corruption_level", 0)),
    )


def _mask(value: Any | None, n: int, *, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full((n,), default, dtype=torch.bool)
    return torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(n).clone()


def _group_components(case, boxes: torch.Tensor) -> torch.Tensor:
    result = torch.full((case.n,), -1, dtype=torch.long)
    component = 0
    for row in case.group_membership.detach().cpu().bool():
        members = set(torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        adjacency = {member: set() for member in members}
        ordered = sorted(members)
        for offset, first in enumerate(ordered):
            for second in ordered[offset + 1 :]:
                if edge_connected(boxes[first], boxes[second], tol=0.0):
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        unseen = set(members)
        while unseen:
            start = min(unseen)
            stack = [start]
            unseen.remove(start)
            while stack:
                current = stack.pop()
                result[current] = component
                neighbors = adjacency[current].intersection(unseen, members)
                unseen.difference_update(neighbors)
                stack.extend(sorted(neighbors, reverse=True))
            component += 1
    return result


def _mib_shape_classes(case, boxes: torch.Tensor) -> torch.Tensor:
    result = torch.full((case.n,), -1, dtype=torch.long)
    scale = case.scale if case.normalized else 1.0
    keys = mib_shape_keys(boxes, scale=scale)
    for row in case.mib_membership.detach().cpu().bool():
        members = torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        class_by_key = {
            key: index for index, key in enumerate(sorted({keys[i] for i in members}))
        }
        for member in members:
            result[member] = class_by_key[keys[member]]
    return result
