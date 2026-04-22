from __future__ import annotations

import torch

from puzzleplace.actions.executor import ExecutionState
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.feedback import (
    build_advantage_dataset_from_cases,
    compute_step_feedback,
    run_advantage_weighted_bc,
)


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="feedback-1",
        block_count=2,
        area_targets=torch.tensor([6.0, 6.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((2, 5)),
        target_positions=torch.tensor([[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 3.0, 2.0]]),
        metrics=None,
    )


def test_step_feedback_prefers_teacher_action_over_shifted_negative() -> None:
    case = _make_case()
    action = TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=0,
        x=0.0,
        y=0.0,
        w=2.0,
        h=3.0,
    )
    feedback = compute_step_feedback(case, ExecutionState(), action)
    assert feedback.positive_legal is True
    assert feedback.advantage > 0
    assert feedback.weight > 1.0


def test_advantage_weighted_bc_reduces_weighted_loss() -> None:
    dataset = build_advantage_dataset_from_cases([_make_case()], max_traces_per_case=2)
    _policy, summary = run_advantage_weighted_bc(
        dataset,
        hidden_dim=32,
        lr=1e-2,
        epochs=20,
        seed=0,
    )
    assert summary.final_loss < summary.initial_loss
    assert summary.mean_weight > 0
