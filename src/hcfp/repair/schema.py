"""Typed intermediate records shared by repair experts and decoders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from hcfp.case import FloorplanCase


Tensor = torch.Tensor


class ExpertKind(str, Enum):
    CONTACT = "contact"
    BOUNDARY = "boundary"
    MIB = "mib"
    TOPOLOGY = "topology"


@dataclass(frozen=True)
class RepairState:
    case: FloorplanCase
    placement: Tensor
    geometry_observed: Tensor
    repair_target: Tensor
    position_mobility: Tensor
    shape_mobility: Tensor
    contact_edges: Tensor
    group_component_id: Tensor
    boundary_missing: Tensor
    mib_shape_class: Tensor
    round_index: int = 0
    corruption_kind: str | None = None
    corruption_level: int = 0

    def __post_init__(self) -> None:
        n = self.case.n
        _shape("placement", self.placement, (n, 4))
        for name in (
            "geometry_observed",
            "repair_target",
            "position_mobility",
            "shape_mobility",
            "group_component_id",
            "boundary_missing",
            "mib_shape_class",
        ):
            _shape(name, getattr(self, name), (n,))
        _shape("contact_edges", self.contact_edges, (None, 4))
        if self.placement.dtype != torch.float32 or self.placement.device.type != "cpu":
            raise ValueError("placement must be a CPU float32 tensor")
        if not bool(torch.isfinite(self.placement).all()) or not bool(
            (self.placement[:, 2:4] > 0).all()
        ):
            raise ValueError("placement must be finite with positive dimensions")
        expected_position = ~self.case.preplaced_mask.detach().cpu().bool()
        expected_shape = ~(
            self.case.fixed_mask.detach().cpu().bool()
            | self.case.preplaced_mask.detach().cpu().bool()
        )
        if not torch.equal(self.position_mobility.detach().cpu().bool(), expected_position):
            raise ValueError("position mobility must be false exactly for preplaced blocks")
        if not torch.equal(self.shape_mobility.detach().cpu().bool(), expected_shape):
            raise ValueError("shape mobility must be false exactly for fixed/preplaced blocks")
        if self.round_index < 0 or self.corruption_level < 0:
            raise ValueError("round and corruption level must be non-negative")


@dataclass(frozen=True)
class RepairObligation:
    expert: ExpertKind
    obligation_id: str
    target_ids: tuple[int, ...]
    relation: str = ""
    debt: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", _ids("target_ids", self.target_ids))
        object.__setattr__(self, "relation", self.relation.strip().upper())
        if not self.obligation_id:
            raise ValueError("obligation_id must be non-empty")
        if self.debt <= 0:
            raise ValueError("obligation debt must be positive")


@dataclass(frozen=True)
class RepairAction:
    expert: ExpertKind
    obligation_id: str
    target_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...] = ()
    relation: str = ""
    shape_spec: tuple[float, float] | None = None
    patch_budget: int = 1
    score: float = 0.0
    corruption_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", _ids("target_ids", self.target_ids))
        object.__setattr__(self, "anchor_ids", _ids("anchor_ids", self.anchor_ids))
        relation = self.relation.strip().upper()
        relation = {"ABOVE": "TOP", "BELOW": "BOTTOM"}.get(relation, relation)
        object.__setattr__(self, "relation", relation)
        if not self.obligation_id:
            raise ValueError("obligation_id must be non-empty")
        if self.patch_budget <= 0:
            raise ValueError("patch_budget must be positive")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.shape_spec is not None:
            shape = tuple(float(value) for value in self.shape_spec)
            if len(shape) != 2 or any(not math.isfinite(value) or value <= 0 for value in shape):
                raise ValueError("shape_spec must contain two finite positive values")
            object.__setattr__(self, "shape_spec", shape)


@dataclass(frozen=True)
class RepairCandidate:
    action: RepairAction
    placement: Tensor
    decoder: str

    def __post_init__(self) -> None:
        _shape("candidate placement", self.placement, (None, 4))
        if self.placement.dtype != torch.float32 or self.placement.device.type != "cpu":
            raise ValueError("candidate placement must be a CPU float32 tensor")
        if not self.decoder:
            raise ValueError("decoder must be non-empty")
        if not bool(torch.isfinite(self.placement).all()) or not bool(
            (self.placement[:, 2:4] > 0).all()
        ):
            raise ValueError("candidate placement must be finite with positive dimensions")


@dataclass(frozen=True)
class RepairOutcome:
    candidate_sha256: str
    accepted: bool
    hard_feasible: bool
    debt_before: int
    debt_after: int
    failure_reason: str | None = None
    cost_before: float | None = None
    cost_after: float | None = None

    def __post_init__(self) -> None:
        if len(self.candidate_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.candidate_sha256
        ):
            raise ValueError("candidate_sha256 must be lowercase SHA-256")
        if self.debt_before < 0 or self.debt_after < 0:
            raise ValueError("repair debt must be non-negative")
        if self.accepted and not self.hard_feasible:
            raise ValueError("accepted repair outcomes must be hard feasible")
        for name in ("cost_before", "cost_after"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")


@dataclass(frozen=True)
class RepairReplayRecord:
    source_id: str
    source_split: str
    split_version: str
    state: RepairState
    obligation: RepairObligation
    action: RepairAction
    candidate: RepairCandidate
    outcome: RepairOutcome

    def __post_init__(self) -> None:
        if not self.source_id or not self.split_version:
            raise ValueError("source_id and split_version must be non-empty")
        if self.source_split not in {"train", "heldout"}:
            raise ValueError("source_split must be train or heldout")
        if self.obligation.expert != self.action.expert:
            raise ValueError("obligation and action experts must match")
        if self.obligation.obligation_id != self.action.obligation_id:
            raise ValueError("obligation and action IDs must match")
        if self.candidate.action != self.action:
            raise ValueError("candidate action must match replay action")
        if self.candidate.placement.shape[0] != self.state.case.n:
            raise ValueError("candidate placement block count must match state")
        if any(index >= self.state.case.n for index in self.action.target_ids + self.action.anchor_ids):
            raise ValueError("action block index is outside the repair state")


def _ids(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative block IDs")
    return result


def _shape(name: str, tensor: Tensor, expected: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(expected) or any(
        size is not None and tensor.shape[index] != size
        for index, size in enumerate(expected)
    ):
        shape = "x".join("*" if size is None else str(size) for size in expected)
        raise ValueError(f"{name} must have shape [{shape}]")
