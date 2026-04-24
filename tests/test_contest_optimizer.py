# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTEST_ROOT = ROOT / "external" / "FloorSet" / "iccad2026contest"
for path in (ROOT, ROOT / "src", CONTEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import ActionPrimitive, ExecutionState, generate_candidate_actions
from puzzleplace.optimizer import ContestOptimizer, contest_case_from_inputs


def test_contest_case_adapter_preserves_block_count_and_targets() -> None:
    target_positions = torch.tensor([[0.0, 0.0, 2.0, 3.0], [-1.0, -1.0, -1.0, -1.0]])
    case = contest_case_from_inputs(
        2,
        torch.tensor([6.0, 6.0]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.empty((0, 3)),
        torch.empty((0, 2)),
        torch.zeros((2, 5)),
        target_positions,
    )
    assert case.block_count == 2
    assert case.target_positions is not None


def test_contest_optimizer_returns_complete_position_list() -> None:
    optimizer = ContestOptimizer()
    positions = optimizer.solve(
        2,
        torch.tensor([6.0, 6.0]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.empty((0, 3)),
        torch.empty((0, 2)),
        torch.zeros((2, 5)),
        torch.tensor([[0.0, 0.0, 2.0, 3.0], [-1.0, -1.0, -1.0, -1.0]]),
    )
    assert len(positions) == 2
    assert all(width > 0 and height > 0 for _x, _y, width, height in positions)


def test_frozen_preplaced_anchor_is_not_reoffered_as_freeze_candidate() -> None:
    case = contest_case_from_inputs(
        2,
        torch.tensor([6.0, 6.0]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.empty((0, 3)),
        torch.empty((0, 2)),
        torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 2.0, 3.0], [-1.0, -1.0, -1.0, -1.0]]),
    )
    state = ExecutionState(
        placements={0: (0.0, 0.0, 2.0, 3.0)},
        frozen_blocks={0},
    )
    candidates = generate_candidate_actions(case, state, remaining_blocks=[1])
    assert not any(
        candidate.primitive is ActionPrimitive.FREEZE and candidate.block_index == 0
        for candidate in candidates
    )


def test_objective_selection_hook_chooses_best_finalized_candidate(monkeypatch) -> None:
    from types import SimpleNamespace

    import puzzleplace.optimizer.contest as contest_mod

    bad_policy = object()
    good_policy = object()
    optimizer = ContestOptimizer(objective_selection_k=2)
    optimizer._candidate_policies = lambda: [("bad", bad_policy), ("good", good_policy)]  # type: ignore[method-assign]

    def fake_semantic_rollout(case, policy, *, role_evidence=None):
        del case, role_evidence
        if policy is bad_policy:
            proposed = {0: (0.0, 0.0, 2.0, 2.0), 1: (100.0, 0.0, 2.0, 2.0)}
        else:
            proposed = {0: (0.0, 0.0, 2.0, 2.0), 1: (2.0, 0.0, 2.0, 2.0)}
        return SimpleNamespace(
            proposed_positions=proposed,
            semantic_completed=True,
            semantic_placed_fraction=1.0,
            fallback_fraction=0.0,
        )

    monkeypatch.setattr(contest_mod, "semantic_rollout", fake_semantic_rollout)

    positions, report = optimizer.solve_with_report(
        2,
        torch.tensor([4.0, 4.0]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.empty((0, 3)),
        torch.empty((0, 2)),
        torch.zeros((2, 5)),
    )

    assert positions == [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    assert report["objective_selection_used"] is True
    assert report["selected_candidate_source"] == "good"
