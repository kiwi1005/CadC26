from __future__ import annotations

import pytest
import torch

from hcfp.constraints.contact_tree import (
    BOTTOM,
    LEFT,
    RIGHT,
    TOP,
    contact_tree_report,
    extract_contacts,
)


def test_extracts_side_specific_contacts_with_positive_overlap() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 2.0, 2.0),
            (2.0, 0.5, 1.0, 1.0),
            (-1.0, 0.25, 1.0, 1.0),
            (0.25, 2.0, 1.0, 1.0),
            (0.5, -1.0, 1.0, 1.0),
            (2.0, 2.0, 1.0, 1.0),
        )
    )

    contacts = extract_contacts(boxes, tolerance=0.0)
    labels = {(edge.first, edge.second): (edge.first_side, edge.second_side) for edge in contacts}

    assert labels[(0, 1)] == (RIGHT, LEFT)
    assert labels[(0, 2)] == (LEFT, RIGHT)
    assert labels[(0, 3)] == (TOP, BOTTOM)
    assert labels[(0, 4)] == (BOTTOM, TOP)
    assert (0, 5) not in labels


def test_contact_tolerance_allows_small_gap_and_requires_overlap() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.00005, 0.25, 1.0, 1.0),
            (2.1, 1.25, 1.0, 1.0),
        )
    )

    assert not extract_contacts(boxes, tolerance=1.0e-5)
    contacts = extract_contacts(boxes, tolerance=1.0e-4)

    assert len(contacts) == 1
    assert (contacts[0].first, contacts[0].second) == (0, 1)
    assert contacts[0].length == pytest.approx(0.75)


def test_contact_tree_fails_closed_or_reports_disconnected_groups() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (5.0, 0.0, 1.0, 1.0),
        )
    )
    groups = torch.tensor(((True, True, True),))

    with pytest.raises(ValueError, match="disconnected contact graph"):
        contact_tree_report(boxes, groups, tolerance=0.0)

    report = contact_tree_report(boxes, groups, tolerance=0.0, fail_on_disconnected=False)
    assert report.disconnected_groups == (0,)
    assert not report.trees[0].connected
    assert [(edge.first, edge.second) for edge in report.trees[0].edges] == [(0, 1)]


def test_contact_tree_is_deterministic_on_weighted_ties() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        )
    )
    groups = torch.tensor(((True, True, True, True),))
    weights = torch.zeros((4, 4))

    first = contact_tree_report(boxes, groups, net_weight=weights, tolerance=0.0)
    second = contact_tree_report(boxes, groups, net_weight=weights, tolerance=0.0)
    edges = [(edge.first, edge.second, edge.first_side) for edge in first.trees[0].edges]

    assert edges == [(0, 1, RIGHT), (0, 2, TOP), (1, 3, TOP)]
    assert edges == [(edge.first, edge.second, edge.first_side) for edge in second.trees[0].edges]


def test_net_weight_and_distance_terms_rank_tree_edges() -> None:
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 2.0),
            (1.0, 0.0, 1.0, 2.0),
            (2.0, 0.0, 1.0, 2.0),
        )
    )
    groups = torch.tensor(((True, True, True),))
    weights = torch.zeros((3, 3))
    weights[1, 2] = weights[2, 1] = 5.0

    report = contact_tree_report(
        boxes,
        groups,
        net_weight=weights,
        tolerance=0.0,
        net_weight_weight=10.0,
        distance_weight=0.5,
    )

    assert [(edge.first, edge.second) for edge in report.trees[0].edges] == [(1, 2), (0, 1)]
    assert report.trees[0].edges[0].sequence_precedence == (1, 2, "horizontal")


def test_singleton_and_empty_groups_are_connected_without_edges() -> None:
    boxes = torch.tensor(((0.0, 0.0, 1.0, 1.0), (3.0, 0.0, 1.0, 1.0)))
    groups = torch.tensor(((False, False), (True, False)))

    report = contact_tree_report(boxes, groups, tolerance=0.0)

    assert report.disconnected_groups == ()
    assert [(tree.members, tree.edges, tree.connected) for tree in report.trees] == [
        ((), (), True),
        ((0,), (), True),
    ]
