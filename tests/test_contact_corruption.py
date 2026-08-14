from __future__ import annotations

from itertools import permutations

import torch

from hcfp.case import from_official
from hcfp.repair.actions import action_sha256
from hcfp.repair.corruption.contact import (
    contact_c2_eligible,
    generate_contact_corruptions,
)
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
        assert (
            tuple(action_sha256(action) for action in rank_contact_actions(order))
            == expected
        )


def test_contact_c2_reslices_closed_patch_and_repairs_without_clean_geometry() -> None:
    constraints = torch.tensor(((0, 0, 0, 1, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0)))
    case = from_official(3, torch.ones(3), [], [], [], constraints)
    clean = torch.tensor(
        ((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (0.0, 2.0, 1.0, 1.0)),
        dtype=torch.float64,
    )
    verify_case = {
        "normalized": False,
        "area_targets": torch.ones(3),
        "constraints": constraints,
        "fixed_mask": case.fixed_mask,
        "preplaced_mask": case.preplaced_mask,
        "group_membership": case.group_membership,
        "mib_membership": case.mib_membership,
    }

    assert contact_c2_eligible(case, clean)
    corruption = generate_contact_corruptions(
        case,
        clean,
        verify_case=verify_case,
        kinds=("C2",),
    )[0]

    assert corruption.kind == "C2"
    assert corruption.debt_after > corruption.debt_before
    assert corruption.decoded_debt < corruption.debt_after
    assert not torch.equal(corruption.placement, clean)
    decoded = decode_contact_action(
        case,
        corruption.placement,
        corruption.inverse_action,
        verify_case=verify_case,
    )
    assert decoded.succeeded
    assert not torch.equal(decoded.placement, clean)


def test_contact_c2_requires_a_bridge_contact() -> None:
    constraints = torch.tensor(((0, 0, 0, 1, 0),) * 3 + ((0, 0, 0, 0, 0),))
    case = from_official(4, torch.tensor((2.0, 1.0, 1.0, 1.0)), [], [], [], constraints)
    clean = torch.tensor(
        (
            (0.0, 0.0, 1.0, 2.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (3.0, 0.0, 1.0, 1.0),
        ),
        dtype=torch.float64,
    )

    assert not contact_c2_eligible(case, clean)
