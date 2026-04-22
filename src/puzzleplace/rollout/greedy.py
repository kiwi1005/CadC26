from __future__ import annotations

from dataclasses import dataclass

import torch

from puzzleplace.actions import ActionExecutor, ExecutionState, generate_candidate_actions
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data import FloorSetCase
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.models.policy import DecoderOutput, TypedActionPolicy
from puzzleplace.roles import WeakRoleEvidence, label_case_roles


@dataclass(slots=True)
class RolloutResult:
    actions: list[TypedAction]
    placements: dict[int, tuple[float, float, float, float]]
    placed_count: int
    all_blocks_placed: bool
    stopped_reason: str
    total_score: float
    feasible: bool | None


def _geometry_vector(candidate: TypedAction) -> torch.Tensor | None:
    if candidate.primitive is ActionPrimitive.PLACE_ABSOLUTE and None not in (
        candidate.x,
        candidate.y,
        candidate.w,
        candidate.h,
    ):
        return torch.tensor(
            [candidate.x, candidate.y, candidate.w, candidate.h], dtype=torch.float32
        )
    return None


def score_action_with_policy(output: DecoderOutput, action: TypedAction) -> float:
    score = float(output.primitive_logits[list(ActionPrimitive).index(action.primitive)].item())
    score += float(output.block_logits[action.block_index].item())
    if action.target_index is not None:
        score += float(output.target_logits[action.block_index, action.target_index].item())
    if action.primitive is ActionPrimitive.ALIGN_BOUNDARY and action.boundary_code is not None:
        boundary_map = {1: 1, 2: 2, 4: 3, 8: 4}
        score += float(
            output.boundary_logits[
                action.block_index, boundary_map.get(int(action.boundary_code), 0)
            ].item()
        )
    geom = _geometry_vector(action)
    if geom is not None:
        pred = output.geometry[action.block_index]
        score -= float(torch.nn.functional.mse_loss(pred, geom).item())
    return score


def greedy_rollout(
    case: FloorSetCase,
    policy: TypedActionPolicy,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    initial_state: ExecutionState | None = None,
    max_steps: int | None = None,
) -> RolloutResult:
    role_evidence = role_evidence or label_case_roles(case)
    state = (
        ExecutionState(
            placements=dict(initial_state.placements),
            frozen_blocks=set(initial_state.frozen_blocks),
        )
        if initial_state is not None
        else ExecutionState()
    )
    executor = ActionExecutor(case)
    actions: list[TypedAction] = []
    total_score = 0.0
    max_steps = max_steps or (case.block_count * 3)

    for _step in range(max_steps):
        remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
        if not remaining:
            break
        output = policy(case, role_evidence=role_evidence, placements=state.placements)
        candidates = generate_candidate_actions(case, state, remaining_blocks=remaining)
        if not candidates:
            return RolloutResult(
                actions,
                dict(state.placements),
                len(state.placements),
                False,
                "no_legal_candidates",
                total_score,
                False,
            )
        scored = [
            (score_action_with_policy(output, candidate), candidate) for candidate in candidates
        ]
        score, chosen = max(scored, key=lambda item: item[0])
        executor.apply(state, chosen)
        actions.append(chosen)
        total_score += float(score)

    all_blocks_placed = len(state.placements) == case.block_count
    feasible = None
    if all_blocks_placed:
        summary = summarize_hard_legality(
            case, [state.placements[idx] for idx in range(case.block_count)]
        )
        feasible = summary.is_feasible
    return RolloutResult(
        actions=actions,
        placements=dict(state.placements),
        placed_count=len(state.placements),
        all_blocks_placed=all_blocks_placed,
        stopped_reason="completed" if all_blocks_placed else "step_limit",
        total_score=total_score,
        feasible=feasible,
    )
