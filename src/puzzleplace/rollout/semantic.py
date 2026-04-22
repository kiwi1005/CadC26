from __future__ import annotations

from dataclasses import dataclass

from puzzleplace.actions import ActionExecutor, ExecutionState, generate_candidate_actions
from puzzleplace.actions.candidates import _default_dims
from puzzleplace.actions.masks import CandidateMode
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.eval.violation import ViolationProfile, summarize_violation_profile
from puzzleplace.models.policy import TypedActionPolicy
from puzzleplace.roles import WeakRoleEvidence, label_case_roles


@dataclass(slots=True)
class SemanticRolloutStep:
    step: int
    primitive: str
    before_semantic_placed: int
    after_semantic_placed: int
    progress_made: bool
    overlap_pairs: int
    total_overlap_area: float


@dataclass(slots=True)
class SemanticRolloutResult:
    actions: list[TypedAction]
    proposed_positions: dict[int, tuple[float, float, float, float]]
    semantic_completed: bool
    semantic_placed_fraction: float
    violation_profile: ViolationProfile
    steps: list[SemanticRolloutStep]
    stopped_reason: str
    fallback_fraction: float


def _bbox(
    positions: dict[int, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not positions:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(x for x, _y, _w, _h in positions.values())
    min_y = min(y for _x, y, _w, _h in positions.values())
    max_x = max(x + width for x, _y, width, _h in positions.values())
    max_y = max(y + height for _x, y, _w, height in positions.values())
    return (min_x, min_y, max_x, max_y)


def _seed_first_action(
    case: FloorSetCase,
    remaining: list[int],
) -> TypedAction:
    preplaced = [
        idx for idx in remaining if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
    ]
    if preplaced and case.target_positions is not None:
        block_index = preplaced[0]
        x, y, width, height = [float(v) for v in case.target_positions[block_index].tolist()]
        return TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=block_index,
            x=max(x, 0.0),
            y=max(y, 0.0),
            w=width,
            h=height,
            metadata={"intent_type": "preplaced_anchor"},
        )

    boundary = [
        idx
        for idx in remaining
        if int(case.constraints[idx, ConstraintColumns.BOUNDARY].item()) != 0
    ]
    if boundary:
        block_index = boundary[0]
    else:
        degrees = [0.0 for _ in range(case.block_count)]
        for src, dst, weight in case.b2b_edges.tolist():
            if src != -1 and int(src) < case.block_count:
                degrees[int(src)] += float(weight)
            if dst != -1 and int(dst) < case.block_count:
                degrees[int(dst)] += float(weight)
        if remaining:
            block_index = max(
                remaining,
                key=lambda idx: (
                    degrees[idx],
                    float(case.area_targets[idx].item()),
                ),
            )
        else:
            block_index = 0
    width, height = _default_dims(case, block_index)
    return TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=block_index,
        x=0.0,
        y=0.0,
        w=width,
        h=height,
        metadata={"intent_type": "seed_anchor"},
    )


def _semantic_heuristic_score(action: TypedAction) -> float:
    source = str(action.metadata.get("source", ""))
    intent = str(action.metadata.get("intent_type", ""))
    score = 0.0
    if source == "preplaced_anchor":
        score += 10.0
    if "pin" in intent:
        score += 3.0
    if "attach" in intent:
        score += 4.0
    if "boundary" in intent:
        score += 3.0
    if source == "semantic_strip":
        score += 1.0
    score -= float(action.metadata.get("total_overlap_area", 0.0)) * 0.001
    score -= float(action.metadata.get("boundary_distance", 0.0)) * 0.001
    score -= float(action.metadata.get("connectivity_proxy_cost", 0.0)) * 0.0001
    return score


def _score_action(
    case: FloorSetCase,
    policy: TypedActionPolicy | None,
    role_evidence: list[WeakRoleEvidence],
    state: ExecutionState,
    action: TypedAction,
) -> float:
    if policy is None:
        return _semantic_heuristic_score(action)
    output = policy(case, role_evidence=role_evidence, placements=state.placements)
    primitive_score = float(
        output.primitive_logits[list(ActionPrimitive).index(action.primitive)].item()
    )
    block_score = float(output.block_logits[action.block_index].item())
    return primitive_score + block_score + _semantic_heuristic_score(action)


def _forced_progress_action(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
) -> TypedAction:
    _min_x, _min_y, max_x, _max_y = _bbox(state.placements)
    width, height = _default_dims(case, block_index)
    return TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=block_index,
        x=max_x + 1.0,
        y=0.0,
        w=width,
        h=height,
        metadata={"source": "semantic_strip", "intent_type": "place_relative_strip"},
    )


def semantic_rollout(
    case: FloorSetCase,
    policy: TypedActionPolicy | None = None,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    max_steps: int | None = None,
    candidate_mode: CandidateMode = "semantic",
) -> SemanticRolloutResult:
    role_evidence = role_evidence or label_case_roles(case)
    state = ExecutionState(last_rollout_mode=candidate_mode)
    executor = ActionExecutor(case)
    max_steps = max_steps or (case.block_count * 4)
    steps: list[SemanticRolloutStep] = []
    forced_actions = 0

    seed = _seed_first_action(
        case,
        [idx for idx in range(case.block_count) if idx not in state.semantic_placed],
    )
    executor.apply(state, seed)

    no_progress = 0
    while len(state.semantic_placed) < case.block_count and state.step < max_steps:
        remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
        before = len(state.semantic_placed)
        candidates = generate_candidate_actions(
            case,
            state,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        if not candidates:
            chosen = _forced_progress_action(case, state, remaining[0])
            forced_actions += 1
        else:
            chosen = max(
                candidates,
                key=lambda action: _score_action(
                    case,
                    policy,
                    role_evidence,
                    state,
                    action,
                ),
            )
            if no_progress >= 2 and chosen.block_index in state.semantic_placed:
                chosen = _forced_progress_action(case, state, remaining[0])
                forced_actions += 1

        executor.apply(state, chosen)
        profile = summarize_violation_profile(case, state.placements)
        after = len(state.semantic_placed)
        progress = after > before
        no_progress = 0 if progress else no_progress + 1
        steps.append(
            SemanticRolloutStep(
                step=state.step,
                primitive=str(chosen.primitive),
                before_semantic_placed=before,
                after_semantic_placed=after,
                progress_made=progress,
                overlap_pairs=profile.overlap_pairs,
                total_overlap_area=profile.total_overlap_area,
            )
        )

    final_profile = summarize_violation_profile(case, state.placements)
    return SemanticRolloutResult(
        actions=list(state.history),
        proposed_positions=dict(state.proposed_positions),
        semantic_completed=len(state.semantic_placed) == case.block_count,
        semantic_placed_fraction=len(state.semantic_placed) / max(case.block_count, 1),
        violation_profile=final_profile,
        steps=steps,
        stopped_reason=(
            "completed" if len(state.semantic_placed) == case.block_count else "max_steps"
        ),
        fallback_fraction=forced_actions / max(len(state.history), 1),
    )
