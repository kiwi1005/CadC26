from __future__ import annotations

from dataclasses import dataclass

from puzzleplace.actions import ActionExecutor, ExecutionState, generate_candidate_actions
from puzzleplace.data import FloorSetCase
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.models.policy import TypedActionPolicy
from puzzleplace.roles import WeakRoleEvidence, label_case_roles

from .greedy import RolloutResult, score_action_with_policy


@dataclass(slots=True)
class _BeamState:
    score: float
    state: ExecutionState
    actions: list


def _clone_state(state: ExecutionState) -> ExecutionState:
    return ExecutionState(placements=dict(state.placements), frozen_blocks=set(state.frozen_blocks))


def beam_rollout(
    case: FloorSetCase,
    policy: TypedActionPolicy,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    beam_width: int = 4,
    per_state_candidates: int = 3,
    max_steps: int | None = None,
) -> RolloutResult:
    role_evidence = role_evidence or label_case_roles(case)
    max_steps = max_steps or (case.block_count * 3)
    beam = [_BeamState(score=0.0, state=ExecutionState(), actions=[])]

    for _step in range(max_steps):
        expanded: list[_BeamState] = []
        any_remaining = False
        for item in beam:
            remaining = [idx for idx in range(case.block_count) if idx not in item.state.placements]
            if not remaining:
                expanded.append(item)
                continue
            any_remaining = True
            output = policy(case, role_evidence=role_evidence, placements=item.state.placements)
            candidates = generate_candidate_actions(case, item.state, remaining_blocks=remaining)
            scored = sorted(((score_action_with_policy(output, c), c) for c in candidates), key=lambda x: x[0], reverse=True)[:per_state_candidates]
            executor = ActionExecutor(case)
            for score, candidate in scored:
                new_state = _clone_state(item.state)
                executor.apply(new_state, candidate)
                expanded.append(_BeamState(score=item.score + float(score), state=new_state, actions=item.actions + [candidate]))
        if not any_remaining:
            break
        if not expanded:
            break
        beam = sorted(expanded, key=lambda item: item.score, reverse=True)[:beam_width]

    best = max(beam, key=lambda item: (len(item.state.placements), item.score))
    all_blocks_placed = len(best.state.placements) == case.block_count
    feasible = None
    if all_blocks_placed:
        summary = summarize_hard_legality(case, [best.state.placements[idx] for idx in range(case.block_count)])
        feasible = summary.is_feasible
    return RolloutResult(
        actions=best.actions,
        placements=dict(best.state.placements),
        placed_count=len(best.state.placements),
        all_blocks_placed=all_blocks_placed,
        stopped_reason="completed" if all_blocks_placed else "beam_exhausted_or_step_limit",
        total_score=best.score,
        feasible=feasible,
    )
