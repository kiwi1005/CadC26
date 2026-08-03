from __future__ import annotations

import torch

from hcfp.constraints.raw_repair import repair_raw_constraints
from hcfp.verify import grouping_violation, verify_feasible


def _source() -> dict[str, object]:
    return {
        "normalized": False,
        "area_targets": torch.ones(4),
        "constraints": torch.tensor(
            [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
        ),
        "group_membership": torch.tensor(
            [[True, True, True, False]]
        ),
        "mib_membership": torch.zeros((0, 4), dtype=torch.bool),
        "boundary_bits": torch.zeros((4, 4), dtype=torch.bool),
    }


def test_raw_contact_tree_replay_connects_group_without_hard_overlap() -> None:
    source = _source()
    placements = (
        (0.0, 0.0, 1.0, 1.0),
        (3.0, 0.0, 1.0, 1.0),
        (6.0, 0.0, 1.0, 1.0),
        (9.0, 0.0, 1.0, 1.0),
    )
    record = {
        "details": {
            "moves": (
                {"members": (1,), "anchor": 0, "child": 1, "side": "right"},
                {"members": (2,), "anchor": 1, "child": 2, "side": "right"},
            )
        }
    }

    repaired = repair_raw_constraints(source, placements, record)

    assert verify_feasible(source, repaired.placements)
    assert grouping_violation(source, repaired.placements) == 0
    assert repaired.group_edges_applied == 2
    assert repaired.group_edges_rejected == 0


def test_raw_contact_replay_rolls_back_move_that_hits_external_block() -> None:
    source = _source()
    placements = (
        (0.0, 0.0, 1.0, 1.0),
        (3.0, 0.0, 1.0, 1.0),
        (6.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
    )
    record = {
        "details": {
            "moves": (
                {"members": (1,), "anchor": 0, "child": 1, "side": "right"},
            )
        }
    }

    repaired = repair_raw_constraints(source, placements, record)

    assert repaired.placements == placements
    assert repaired.group_edges_applied == 0
    assert repaired.group_edges_rejected == 1


def test_raw_contact_tree_commits_atomically_from_rounding_overlaps() -> None:
    source = _source()
    placements = (
        (0.0, 0.0, 1.0, 1.0),
        (0.99999, 0.0, 1.0, 1.0),
        (1.99998, 0.0, 1.0, 1.0),
        (9.0, 0.0, 1.0, 1.0),
    )
    record = {
        "details": {
            "moves": (
                {"members": (1,), "anchor": 0, "child": 1, "side": "right"},
                {"members": (2,), "anchor": 1, "child": 2, "side": "right"},
            )
        }
    }

    assert not verify_feasible(source, placements)

    repaired = repair_raw_constraints(source, placements, record)

    assert verify_feasible(source, repaired.placements)
    assert grouping_violation(source, repaired.placements) == 0
    assert repaired.group_edges_applied == 2
    assert repaired.group_edges_rejected == 0
