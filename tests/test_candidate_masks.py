from __future__ import annotations

import torch

from puzzleplace.actions import (
    ExecutionState,
    TypedAction,
    check_action_mask,
    compute_expert_candidate_coverage,
    generate_candidate_actions,
)
from puzzleplace.actions.schema import ActionPrimitive
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.trajectory import generate_pseudo_traces


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="cand-1",
        block_count=3,
        area_targets=torch.tensor([6.0, 6.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 3.0],
                [2.0, 0.0, 3.0, 2.0],
                [0.0, 3.0, 2.0, 2.0],
            ]
        ),
        metrics=torch.tensor([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
    )


def test_mask_rejects_overlap_and_accepts_valid_absolute() -> None:
    case = _make_case()
    state = ExecutionState(placements={0: (0.0, 0.0, 2.0, 3.0)})
    legal = TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=2, x=0.0, y=3.0, w=2.0, h=2.0)
    illegal = TypedAction(ActionPrimitive.PLACE_ABSOLUTE, block_index=2, x=1.0, y=1.0, w=2.0, h=2.0)
    assert check_action_mask(case, state, legal).allowed is True
    assert check_action_mask(case, state, illegal).allowed is False


def test_candidate_generator_returns_legal_candidates() -> None:
    case = _make_case()
    state = ExecutionState(placements={0: (0.0, 0.0, 2.0, 3.0)})
    candidates = generate_candidate_actions(case, state, remaining_blocks=[1, 2])
    assert candidates
    assert all(check_action_mask(case, state, action).allowed for action in candidates)


def test_expert_candidate_coverage_reports_augmented_hits() -> None:
    case = _make_case()
    traces = generate_pseudo_traces(case, max_traces=2)
    report = compute_expert_candidate_coverage(case, traces)
    assert report.total_steps > 0
    assert report.augmented_coverage == 1.0
