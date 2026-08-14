from __future__ import annotations

import json

import pytest

from hcfp.repair.replay import (
    candidate_sha256,
    repair_replay_dumps,
    repair_replay_loads,
    repair_replay_to_payload,
)
from hcfp.repair.schema import (
    ExpertKind,
    RepairAction,
    RepairCandidate,
    RepairObligation,
    RepairOutcome,
    RepairReplayRecord,
)
from hcfp.repair.state import build_repair_state, state_to_payload
from test_repair_schema import repair_fixture


def _record() -> RepairReplayRecord:
    case, placement = repair_fixture()
    state = build_repair_state(
        case, placement, repair_target=(False, False, True, False)
    )
    obligation = RepairObligation(ExpertKind.CONTACT, "group:1", (0, 1, 2), debt=1)
    action = RepairAction(
        ExpertKind.CONTACT,
        "group:1",
        (2,),
        (1,),
        "LEFT",
        patch_budget=3,
        score=0.75,
        corruption_id="contact-c1:7",
    )
    candidate = RepairCandidate(action, placement.clone(), "contact-v1")
    outcome = RepairOutcome(
        candidate_sha256(candidate),
        accepted=True,
        hard_feasible=True,
        debt_before=1,
        debt_after=0,
        cost_before=7.0,
        cost_after=6.5,
    )
    return RepairReplayRecord(
        "worker_0/layouts_0.th:0",
        "train",
        "ccrl-source-v1",
        state,
        placement.double(),
        obligation,
        action,
        candidate,
        outcome,
    )


def test_repair_replay_round_trip_and_sha256_are_deterministic() -> None:
    record = _record()
    encoded = repair_replay_dumps(record)
    loaded = repair_replay_loads(encoded)

    assert repair_replay_dumps(loaded) == encoded
    assert state_to_payload(loaded.state) == state_to_payload(record.state)
    assert loaded.decoder_placement.dtype == record.decoder_placement.dtype
    assert loaded.decoder_placement.tolist() == record.decoder_placement.tolist()
    assert loaded.action == record.action
    assert len(repair_replay_to_payload(record)["record_sha256"]) == 64

    reordered = json.dumps(json.loads(encoded), sort_keys=False)
    assert repair_replay_dumps(repair_replay_loads(reordered)) == encoded


def test_repair_replay_rejects_tampering() -> None:
    payload = repair_replay_to_payload(_record())
    payload["outcome"]["debt_after"] = 2
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        repair_replay_loads(json.dumps(payload))

    payload = repair_replay_to_payload(_record())
    payload["decoder_placement"][0][0] += 1.0
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        repair_replay_loads(json.dumps(payload))

    payload = repair_replay_to_payload(_record())
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema version"):
        repair_replay_loads(json.dumps(payload))
