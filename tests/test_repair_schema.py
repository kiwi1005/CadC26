from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.data import inverse_transform
from hcfp.geometry import normalize_xywh
from hcfp.repair.actions import action_sha256, transform_action
from hcfp.repair.schema import ExpertKind, RepairAction
from hcfp.repair.state import build_repair_state
from hcfp.constraints.contact_tree import extract_contacts
from hcfp.verify import boundary_missing, grouping_violation, mib_violation


def repair_fixture():
    constraints = torch.tensor(
        (
            (1, 0, 1, 1, 1),
            (0, 1, 1, 1, 0),
            (0, 0, 1, 1, 0),
            (0, 0, 0, 0, 0),
        )
    )
    targets = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0, -1.0),
        )
    )
    case = from_official(4, torch.ones(4), [], [], [], constraints, targets)
    raw = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (3.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )
    return case, normalize_xywh(case, raw)


def test_state_reproduces_dynamic_constraints_and_mobility() -> None:
    case, placement = repair_fixture()
    state = build_repair_state(
        case,
        placement,
        geometry_observed=(True, True, False, True),
        repair_target=(False, False, True, False),
        round_index=2,
        corruption_kind="contact_c1",
        corruption_level=1,
    )

    assert state.position_mobility.tolist() == [True, False, True, True]
    assert state.shape_mobility.tolist() == [False, False, True, True]
    assert state.group_component_id.tolist() == [0, 0, 1, -1]
    assert state.mib_shape_class.tolist() == [0, 0, 0, -1]
    assert state.contact_edges[:, :2].tolist() == [[0, 1], [2, 3]]
    assert torch.equal(state.boundary_missing, boundary_missing(case, placement))
    assert grouping_violation(case, placement) == 1
    assert mib_violation(case, placement) == 0
    assert len(state.contact_edges) == len(extract_contacts(placement, tolerance=0.0))
    assert state.geometry_observed.tolist() == [True, True, False, True]
    assert state.repair_target.tolist() == [False, False, True, False]


def test_state_uses_exact_contact_geometry_without_cross_group_id_collisions() -> None:
    case = from_official(
        4,
        torch.tensor((288.0, 308.0, 1.0, 1.0)),
        [],
        [],
        [],
        torch.tensor(
            (
                (0, 0, 0, 1, 0),
                (0, 0, 0, 1, 0),
                (0, 0, 0, 2, 0),
                (0, 0, 0, 2, 0),
            )
        ),
        torch.full((4, 4), -1.0),
    )
    exact = torch.tensor(
        (
            (42.0, 130.0, 16.0, 18.0),
            (42.0, 116.0, 22.0, 14.0),
            (80.0, 0.0, 1.0, 1.0),
            (81.0, 0.0, 1.0, 1.0),
        ),
        dtype=torch.float64,
    )
    normalized = normalize_xywh(case, exact)

    assert not extract_contacts(normalized, tolerance=0.0)
    state = build_repair_state(
        case,
        normalized,
        exact_contact_placement=exact,
    )

    assert state.contact_edges[:, :2].tolist() == [[0, 1], [2, 3]]
    assert state.group_component_id.tolist() == [0, 0, 1, 1]


def test_action_identity_and_d4_round_trip_are_canonical() -> None:
    action = RepairAction(
        expert=ExpertKind.CONTACT,
        obligation_id="group:1",
        target_ids=(3, 1, 3),
        anchor_ids=(2,),
        relation="above",
        shape_spec=(2.0, 4.0),
        patch_budget=4,
        score=0.9,
        corruption_id="c0:1",
    )
    equivalent = RepairAction(
        expert=ExpertKind.CONTACT,
        obligation_id="group:1",
        target_ids=(1, 3),
        anchor_ids=(2,),
        relation="TOP",
        shape_spec=(2.0, 4.0),
        patch_budget=4,
        score=-2.0,
        corruption_id="different-teacher",
    )

    assert action_sha256(action) == action_sha256(equivalent)
    for name in (
        "identity",
        "hflip",
        "vflip",
        "rot90",
        "rot180",
        "rot270",
        "transpose",
        "antitranspose",
    ):
        restored = transform_action(
            transform_action(action, name), inverse_transform(name)
        )
        assert action_sha256(restored) == action_sha256(action)
    rotated = transform_action(action, "rot90")
    assert rotated.relation == "LEFT"
    assert rotated.shape_spec == (4.0, 2.0)
    assert transform_action(action, "transpose").relation == "RIGHT"

    boundary = RepairAction(
        ExpertKind.BOUNDARY,
        "boundary:0:left",
        (0,),
        relation="LEFT_PERIMETER",
    )
    assert transform_action(boundary, "hflip").relation == "RIGHT_PERIMETER"
