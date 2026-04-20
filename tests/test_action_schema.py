from __future__ import annotations

import pytest
import torch

from puzzleplace.actions import ActionPrimitive, TypedAction, replay_actions
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.geometry import summarize_hard_legality


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="action-1",
        block_count=3,
        area_targets=torch.tensor([6.0, 6.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((3, 5)),
        target_positions=None,
        metrics=None,
        raw={"note": "target positions intentionally omitted to prove replay does not depend on them"},
    )


def test_replay_actions_places_blocks_without_target_positions() -> None:
    case = _make_case()
    actions = [
        TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=0, x=0.0, y=0.0, w=2.0, h=3.0),
        TypedAction(ActionPrimitive.PLACE_RELATIVE, block_index=1, target_index=0, dx=0.0, dy=0.0, w=3.0, h=2.0),
        TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=2, x=0.0, y=3.0, w=2.0, h=2.0),
    ]
    state = replay_actions(case, actions)
    assert state.placements[0] == (0.0, 0.0, 2.0, 3.0)
    assert state.placements[1] == (2.0, 0.0, 3.0, 2.0)
    assert state.placements[2] == (0.0, 3.0, 2.0, 2.0)


def test_move_resize_and_freeze_are_enforced() -> None:
    case = _make_case()
    actions = [
        TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=0, x=1.0, y=1.0, w=2.0, h=3.0),
        TypedAction(ActionPrimitive.MOVE, block_index=0, dx=2.0, dy=-1.0),
        TypedAction(ActionPrimitive.RESIZE, block_index=0, w=3.0, h=2.0),
        TypedAction(ActionPrimitive.FREEZE, block_index=0),
    ]
    state = replay_actions(case, actions)
    assert state.placements[0] == (3.0, 0.0, 3.0, 2.0)
    with pytest.raises(ValueError):
        replay_actions(case, [TypedAction(ActionPrimitive.MOVE, block_index=0, dx=1.0, dy=0.0)], initial_state=state)


def test_replayed_positions_can_be_checked_for_legality() -> None:
    case = _make_case()
    positions = replay_actions(
        case,
        [
            TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=0, x=0.0, y=0.0, w=2.0, h=3.0),
            TypedAction(ActionPrimitive.PLACE_RELATIVE, block_index=1, target_index=0, dx=0.0, dy=0.0, w=3.0, h=2.0),
            TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=2, x=0.0, y=3.0, w=2.0, h=2.0),
        ],
    ).placements
    summary = summarize_hard_legality(case, [positions[i] for i in range(case.block_count)])
    assert summary.is_feasible is True
