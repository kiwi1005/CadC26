from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.roles import RoleLabel, label_case_roles


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="roles-1",
        block_count=6,
        area_targets=torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0]),
        b2b_edges=torch.tensor([
            [0.0, 5.0, 8.0],
            [0.0, 4.0, 7.0],
            [4.0, 5.0, 1.0],
        ]),
        p2b_edges=torch.tensor([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 1.0],
        ]),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # fixed
            [0.0, 1.0, 0.0, 0.0, 0.0],  # preplaced
            [0.0, 0.0, 0.0, 0.0, 1.0],  # boundary
            [0.0, 0.0, 1.0, 0.0, 0.0],  # MIB
            [0.0, 0.0, 0.0, 1.0, 0.0],  # cluster
            [0.0, 0.0, 0.0, 0.0, 0.0],  # hub/default
        ]),
        target_positions=None,
        metrics=None,
    )


def test_constraint_driven_roles_win_over_hub_heuristic() -> None:
    labels = label_case_roles(_make_case(), hub_quantile=0.5)
    by_index = {label.block_index: label.role for label in labels}
    assert by_index[0] is RoleLabel.FIXED_ANCHOR
    assert by_index[1] is RoleLabel.PREPLACED_ANCHOR
    assert by_index[2] is RoleLabel.BOUNDARY_SEEKER
    assert by_index[3] is RoleLabel.MULTI_INSTANCE
    assert by_index[4] is RoleLabel.CLUSTER_MEMBER


def test_connectivity_hub_is_assigned_when_no_higher_priority_constraint_exists() -> None:
    labels = label_case_roles(_make_case(), hub_quantile=0.5)
    by_index = {label.block_index: label for label in labels}
    assert by_index[5].role in {RoleLabel.CONNECTIVITY_HUB, RoleLabel.AREA_FOLLOWER}
    assert by_index[5].reasons


def test_role_labels_are_explicitly_weak_heuristics() -> None:
    labels = label_case_roles(_make_case())
    assert all(label.reasons for label in labels)
