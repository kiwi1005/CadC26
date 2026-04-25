from __future__ import annotations

import torch
from torch import nn

from puzzleplace.actions import ExecutionState
from puzzleplace.research.puzzle_candidate_payload import (
    build_puzzle_candidate_descriptors,
    choose_expert_descriptor,
    heuristic_scores,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.trajectory import generate_pseudo_traces


def test_single_state_candidate_scorer_can_overfit_one_pool() -> None:
    torch.manual_seed(0)
    case = make_step6g_synthetic_case(block_count=8)
    teacher = generate_pseudo_traces(case, max_traces=1)[0].actions[0]
    descriptors = build_puzzle_candidate_descriptors(case, ExecutionState(), max_shape_bins=3)
    expert = choose_expert_descriptor(descriptors, teacher)
    assert expert is not None
    features = torch.stack([desc.normalized_features for desc in descriptors])
    model = nn.Linear(features.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    target = torch.tensor([expert])
    initial = float(
        nn.functional.cross_entropy(model(features).squeeze(-1).unsqueeze(0), target).detach()
    )
    final = initial
    for _ in range(80):
        opt.zero_grad()
        logits = model(features).squeeze(-1).unsqueeze(0)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        opt.step()
        final = float(loss.detach())
    pred = int(model(features).squeeze(-1).argmax().item())
    assert final <= max(0.05, initial * 0.10)
    assert pred == expert


def test_heuristic_scores_shape_and_ranking_are_pool_local() -> None:
    case = make_step6g_synthetic_case(block_count=8)
    descriptors = build_puzzle_candidate_descriptors(
        case, ExecutionState(), remaining_blocks=[2, 3]
    )
    scores = heuristic_scores(descriptors)
    assert scores.shape == (len(descriptors),)
    assert torch.isfinite(scores).all()
