from types import SimpleNamespace

import torch

from hcfp.contact_patch import ContactPatchCandidate
from hcfp.contact_policy import (
    CONTACT_FEATURE_NAMES,
    ContactPolicy,
    contact_candidate_features,
    load_contact_policy,
    rank_contact_candidates,
    save_contact_policy,
)


def _candidate() -> ContactPatchCandidate:
    placement = torch.tensor(
        ((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0)),
        dtype=torch.float64,
    )
    return ContactPatchCandidate(
        placement=placement,
        group_index=0,
        bridge_member=0,
        anchor_member=1,
        members=(0, 1),
        side="right",
        grouping_before=1,
        grouping_after=0,
    )


def test_contact_policy_features_and_round_trip(tmp_path) -> None:
    case = SimpleNamespace(
        group_membership=torch.tensor(((True, True, False),)),
        b2b_weight=torch.tensor(
            ((0.0, 3.0, 0.0), (3.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        ),
    )
    raw_case = {"boundary_bits": torch.tensor((1, 0, 0), dtype=torch.int64)}
    placements = torch.tensor(
        ((0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (3.0, 0.0, 1.0, 1.0)),
        dtype=torch.float64,
    )
    candidate = _candidate()

    features = contact_candidate_features(case, raw_case, placements, candidate)
    assert features.shape == (len(CONTACT_FEATURE_NAMES),)
    assert bool(torch.isfinite(features).all())

    policy = ContactPolicy()
    policy.set_normalization(features, torch.ones_like(features))
    path = tmp_path / "contact.pt"
    save_contact_policy(policy, path, metadata={"purpose": "unit-test"})
    loaded, metadata = load_contact_policy(path)
    ranked = rank_contact_candidates(loaded, case, raw_case, placements, (candidate,))
    assert metadata["purpose"] == "unit-test"
    assert len(ranked) == 1
    assert ranked[0][0] == candidate
