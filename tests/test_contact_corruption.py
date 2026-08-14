from __future__ import annotations

from itertools import permutations

import torch

from hcfp.repair.actions import action_sha256
from hcfp.repair.corruption.contact import generate_contact_corruptions
from hcfp.repair.decoders.base import DecodeFailure
from hcfp.repair.decoders.contact import (
    decode_contact_action,
    rank_contact_actions,
)
from hcfp.repair.schema import ExpertKind, RepairAction
from test_repair_dataset import _clean_source


def test_contact_c0_c1_preserve_mobility_and_inverse_reduces_debt() -> None:
    sample, source = _clean_source()
    clean = source["fp_sol_xywh"].double()
    corruptions = generate_contact_corruptions(sample.case, clean, verify_case=source)

    assert {corruption.kind for corruption in corruptions} == {"C0", "C1"}
    for corruption in corruptions:
        assert corruption.debt_after > corruption.debt_before
        assert corruption.decoded_debt < corruption.debt_after
        assert torch.equal(
            corruption.placement[sample.case.preplaced_mask],
            clean[sample.case.preplaced_mask],
        )
        assert torch.equal(
            corruption.placement[sample.case.fixed_mask, 2:4],
            clean[sample.case.fixed_mask, 2:4],
        )


def test_contact_decoder_rejects_preplaced_target_and_rank_is_order_stable() -> None:
    sample, source = _clean_source()
    clean = source["fp_sol_xywh"].double()
    invalid = RepairAction(
        ExpertKind.CONTACT,
        "contact-group:0",
        (0,),
        (1,),
        "LEFT",
        patch_budget=2,
    )
    result = decode_contact_action(sample.case, clean, invalid, verify_case=source)
    assert result.failure == DecodeFailure.IMMOBILE_TARGET

    actions = (
        RepairAction(ExpertKind.CONTACT, "g", (2,), (1,), "LEFT", score=1.0),
        RepairAction(ExpertKind.CONTACT, "g", (3,), (1,), "RIGHT", score=2.0),
    )
    expected = tuple(action_sha256(action) for action in rank_contact_actions(actions))
    for order in permutations(actions):
        assert tuple(action_sha256(action) for action in rank_contact_actions(order)) == expected
