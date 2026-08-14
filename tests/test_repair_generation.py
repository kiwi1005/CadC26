from __future__ import annotations

import json

import pytest

from hcfp.repair.replay import (
    repair_generation_dumps,
    repair_generation_loads,
    repair_generation_to_payload,
)
from hcfp.repair.schema import ExpertKind, RepairAction, RepairGenerationRecord


def _action() -> RepairAction:
    return RepairAction(
        ExpertKind.CONTACT,
        "contact-group:0",
        (2,),
        (1,),
        "RIGHT",
        patch_budget=2,
        corruption_id="contact-c1:fixture",
    )


def test_generation_record_preserves_success_and_acceptable_actions() -> None:
    action = _action()
    record = RepairGenerationRecord(
        "worker_0/layouts_0.th:0",
        "train",
        "ccrl-source-v1",
        "c1",
        True,
        True,
        None,
        action,
        True,
        (action,),
        (action,),
        oracle_action_count=4,
        oracle_best_gain=1.0,
    )

    encoded = repair_generation_dumps(record)
    loaded = repair_generation_loads(encoded)

    assert loaded == record
    assert repair_generation_dumps(loaded) == encoded


def test_generation_record_preserves_failure_category_and_detects_tampering() -> None:
    record = RepairGenerationRecord(
        "worker_0/layouts_0.th:0",
        "heldout",
        "ccrl-source-v1",
        "c2",
        True,
        False,
        "inverse_hard_infeasible",
    )
    payload = repair_generation_to_payload(record)
    payload["generation_failure_reason"] = "no_debt_increase"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        repair_generation_loads(json.dumps(payload))
