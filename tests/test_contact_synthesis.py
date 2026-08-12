from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.case import from_official  # noqa: E402
from hcfp.contact_synthesis import (  # noqa: E402
    apply_contact_obligations,
    synthesize_contact_obligations,
)
from hcfp.geometry import normalize_xywh  # noqa: E402
from hcfp.verify import grouping_violation, verify_feasible  # noqa: E402


def _case(block_count: int, groups: list[list[int]]):
    constraints = [[0, 0, 0, 0, 0] for _ in range(block_count)]
    for group_id, members in enumerate(groups, start=1):
        for member in members:
            constraints[member][3] = group_id
    return from_official(
        block_count,
        torch.ones(block_count),
        [],
        [],
        [],
        constraints,
    )


def test_disconnected_three_component_group_yields_spanning_tree() -> None:
    case = _case(5, [[0, 1, 2, 3, 4]])
    placements = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
            [6.0, 0.0, 1.0, 1.0],
            [10.0, 0.0, 1.0, 1.0],
        ]
    )
    before = placements.clone()

    result = synthesize_contact_obligations(case, placements)

    assert result.component_groups == (((0, 1), (2, 3), (4,)),)
    assert len(result.candidate_edges) == 3
    assert len(result.obligations) == 2
    assert {edge.component_a for edge in result.obligations} | {
        edge.component_b for edge in result.obligations
    } == {(0, 1), (2, 3), (4,)}
    assert all(edge.bridge_member in edge.members for edge in result.obligations)
    assert torch.equal(placements, before)


def test_connected_group_yields_no_obligations() -> None:
    case = _case(4, [[0, 1, 2, 3]])
    placements = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
        ]
    )

    result = synthesize_contact_obligations(case, placements)

    assert result.obligations == ()
    assert result.candidate_edges == ()
    assert result.component_groups == (((0, 1, 2, 3),),)
    assert result.bridge_members == ()


def test_contact_synthesis_is_deterministic() -> None:
    case = _case(5, [[0, 1, 2, 3, 4]])
    placements = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
            [6.0, 0.0, 1.0, 1.0],
            [10.0, 0.0, 1.0, 1.0],
        ]
    )

    first = synthesize_contact_obligations(case, placements)
    second = synthesize_contact_obligations(case, placements)

    assert first == second
    assert {edge.components for edge in first.obligations} == {
        ((0, 1), (2, 3)),
        ((2, 3), (4,)),
    }


def test_apply_contact_obligations_returns_monotone_exact_candidates() -> None:
    case = _case(5, [[0, 1, 2, 3, 4]])
    raw = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [6.0, 0.0, 1.0, 1.0],
            [9.0, 0.0, 1.0, 1.0],
            [12.0, 0.0, 1.0, 1.0],
        ]
    )
    placements = normalize_xywh(case, raw)
    before = grouping_violation(case, placements)

    candidates = apply_contact_obligations(case, placements)

    assert len(candidates) == 4
    previous = before
    for candidate in candidates:
        current = grouping_violation(case, candidate)
        assert current < previous
        assert verify_feasible(case, candidate)
        previous = current
    assert torch.equal(placements, normalize_xywh(case, raw))


def test_apply_contact_obligations_never_moves_preplaced_members() -> None:
    case = from_official(
        4,
        torch.ones(4),
        [],
        [],
        [],
        [
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
    )
    raw = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [6.0, 0.0, 1.0, 1.0],
            [12.0, 0.0, 1.0, 1.0],
        ]
    )
    placements = normalize_xywh(case, raw)
    candidates = apply_contact_obligations(case, placements)

    assert candidates
    for candidate in candidates:
        assert torch.equal(
            candidate[case.preplaced_mask], placements[case.preplaced_mask]
        )
        assert verify_feasible(case, candidate)


def test_apply_contact_obligations_rejects_collision_with_outside_obstacle() -> None:
    case = from_official(
        3,
        [1.0, 1.0, 2.0],
        [],
        [],
        [],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    raw = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 1.0],
        ]
    )
    placements = normalize_xywh(case, raw)

    assert grouping_violation(case, placements) == 1
    assert verify_feasible(case, placements)
    assert apply_contact_obligations(case, placements) == ()
