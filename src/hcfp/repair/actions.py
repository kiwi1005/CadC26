"""Canonical repair-action identity and D4 transforms."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from hcfp.data import D4_TRANSFORMS
from hcfp.repair.schema import ExpertKind, RepairAction


_SIDE_MAP = {
    "identity": {"LEFT": "LEFT", "RIGHT": "RIGHT", "TOP": "TOP", "BOTTOM": "BOTTOM"},
    "hflip": {"LEFT": "RIGHT", "RIGHT": "LEFT", "TOP": "TOP", "BOTTOM": "BOTTOM"},
    "vflip": {"LEFT": "LEFT", "RIGHT": "RIGHT", "TOP": "BOTTOM", "BOTTOM": "TOP"},
    "rot90": {"LEFT": "BOTTOM", "RIGHT": "TOP", "TOP": "LEFT", "BOTTOM": "RIGHT"},
    "rot180": {"LEFT": "RIGHT", "RIGHT": "LEFT", "TOP": "BOTTOM", "BOTTOM": "TOP"},
    "rot270": {"LEFT": "TOP", "RIGHT": "BOTTOM", "TOP": "RIGHT", "BOTTOM": "LEFT"},
    "transpose": {"LEFT": "BOTTOM", "RIGHT": "TOP", "TOP": "RIGHT", "BOTTOM": "LEFT"},
    "antitranspose": {"LEFT": "TOP", "RIGHT": "BOTTOM", "TOP": "LEFT", "BOTTOM": "RIGHT"},
}
_AXIS_SWAPS = {"rot90", "rot270", "transpose", "antitranspose"}


def action_to_payload(action: RepairAction, *, identity_only: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "expert": action.expert.value,
        "obligation_id": action.obligation_id,
        "target_ids": list(action.target_ids),
        "anchor_ids": list(action.anchor_ids),
        "relation": action.relation,
        "shape_spec": list(action.shape_spec) if action.shape_spec is not None else None,
        "patch_budget": action.patch_budget,
    }
    if not identity_only:
        payload.update(score=action.score, corruption_id=action.corruption_id)
    return payload


def action_from_payload(payload: dict[str, Any]) -> RepairAction:
    return RepairAction(
        expert=ExpertKind(payload["expert"]),
        obligation_id=str(payload["obligation_id"]),
        target_ids=tuple(int(value) for value in payload["target_ids"]),
        anchor_ids=tuple(int(value) for value in payload.get("anchor_ids", ())),
        relation=str(payload.get("relation", "")),
        shape_spec=(
            tuple(float(value) for value in payload["shape_spec"])
            if payload.get("shape_spec") is not None
            else None
        ),
        patch_budget=int(payload.get("patch_budget", 1)),
        score=float(payload.get("score", 0.0)),
        corruption_id=(
            str(payload["corruption_id"])
            if payload.get("corruption_id") is not None
            else None
        ),
    )


def action_sha256(action: RepairAction) -> str:
    encoded = json.dumps(
        action_to_payload(action, identity_only=True),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def transform_action(action: RepairAction, name: str) -> RepairAction:
    if name not in D4_TRANSFORMS:
        raise ValueError(f"unknown D4 transform {name!r}")
    relation = _transform_relation(action.relation, name)
    shape = action.shape_spec
    if shape is not None and name in _AXIS_SWAPS:
        shape = (shape[1], shape[0])
    return replace(action, relation=relation, shape_spec=shape)


def _transform_relation(relation: str, name: str) -> str:
    suffix = ""
    base = relation
    if relation.endswith("_PERIMETER"):
        base = relation[: -len("_PERIMETER")]
        suffix = "_PERIMETER"
    if base in _SIDE_MAP[name]:
        return _SIDE_MAP[name][base] + suffix
    if base in {"HORIZONTAL", "VERTICAL"} and name in _AXIS_SWAPS:
        return "VERTICAL" if base == "HORIZONTAL" else "HORIZONTAL"
    return relation
