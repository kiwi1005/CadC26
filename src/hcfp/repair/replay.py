"""Stable JSON serialization for repair-policy replay rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from hcfp.repair.actions import action_from_payload, action_sha256, action_to_payload
from hcfp.repair.schema import (
    ExpertKind,
    RepairCandidate,
    RepairObligation,
    RepairOutcome,
    RepairReplayRecord,
)
from hcfp.repair.state import state_from_payload, state_to_payload


REPAIR_REPLAY_SCHEMA_VERSION = 1


def candidate_to_payload(candidate: RepairCandidate) -> dict[str, Any]:
    return {
        "action": action_to_payload(candidate.action),
        "placement": candidate.placement.tolist(),
        "decoder": candidate.decoder,
    }


def candidate_sha256(candidate: RepairCandidate) -> str:
    return _sha256(
        {
            "action_sha256": action_sha256(candidate.action),
            "placement": candidate.placement.tolist(),
            "decoder": candidate.decoder,
        }
    )


def repair_replay_to_payload(record: RepairReplayRecord) -> dict[str, Any]:
    payload = {
        "schema_version": REPAIR_REPLAY_SCHEMA_VERSION,
        "source_id": record.source_id,
        "source_split": record.source_split,
        "split_version": record.split_version,
        "state": state_to_payload(record.state),
        "obligation": {
            "expert": record.obligation.expert.value,
            "obligation_id": record.obligation.obligation_id,
            "target_ids": list(record.obligation.target_ids),
            "relation": record.obligation.relation,
            "debt": record.obligation.debt,
        },
        "action": action_to_payload(record.action),
        "candidate": candidate_to_payload(record.candidate),
        "outcome": {
            "candidate_sha256": record.outcome.candidate_sha256,
            "accepted": record.outcome.accepted,
            "hard_feasible": record.outcome.hard_feasible,
            "debt_before": record.outcome.debt_before,
            "debt_after": record.outcome.debt_after,
            "failure_reason": record.outcome.failure_reason,
            "cost_before": record.outcome.cost_before,
            "cost_after": record.outcome.cost_after,
        },
    }
    payload["record_sha256"] = _sha256(payload)
    return payload


def repair_replay_from_payload(payload: dict[str, Any]) -> RepairReplayRecord:
    if int(payload.get("schema_version", -1)) != REPAIR_REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported repair replay schema version")
    expected = payload.get("record_sha256")
    raw = {key: value for key, value in payload.items() if key != "record_sha256"}
    if expected != _sha256(raw):
        raise ValueError("repair replay SHA-256 mismatch")
    obligation_payload = payload["obligation"]
    obligation = RepairObligation(
        expert=ExpertKind(obligation_payload["expert"]),
        obligation_id=str(obligation_payload["obligation_id"]),
        target_ids=tuple(int(value) for value in obligation_payload["target_ids"]),
        relation=str(obligation_payload.get("relation", "")),
        debt=int(obligation_payload.get("debt", 1)),
    )
    action = action_from_payload(payload["action"])
    candidate_payload = payload["candidate"]
    candidate = RepairCandidate(
        action=action_from_payload(candidate_payload["action"]),
        placement=torch.as_tensor(candidate_payload["placement"], dtype=torch.float32),
        decoder=str(candidate_payload["decoder"]),
    )
    outcome = RepairOutcome(**payload["outcome"])
    if candidate_sha256(candidate) != outcome.candidate_sha256:
        raise ValueError("repair outcome references a different candidate")
    return RepairReplayRecord(
        source_id=str(payload["source_id"]),
        source_split=str(payload["source_split"]),
        split_version=str(payload["split_version"]),
        state=state_from_payload(payload["state"]),
        obligation=obligation,
        action=action,
        candidate=candidate,
        outcome=outcome,
    )


def repair_replay_dumps(record: RepairReplayRecord) -> str:
    return json.dumps(
        repair_replay_to_payload(record),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def repair_replay_loads(data: str) -> RepairReplayRecord:
    return repair_replay_from_payload(json.loads(data))


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
