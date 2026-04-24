from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from puzzleplace.actions.executor import ActionExecutor, ExecutionState
from puzzleplace.actions.masks import CandidateMode, check_action_mask
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.eval import evaluate_positions
from puzzleplace.geometry.boxes import pairwise_intersection_area
from puzzleplace.models import TypedActionPolicy
from puzzleplace.roles import label_case_roles
from puzzleplace.train import BCStepRecord, compute_bc_loss
from puzzleplace.train.dataset_bc import action_to_targets
from puzzleplace.trajectory import generate_negative_actions, generate_pseudo_traces


@dataclass(slots=True)
class StepFeedback:
    verifier_score: float
    baseline_score: float
    advantage: float
    weight: float
    positive_legal: bool
    baseline_legal: bool | None


@dataclass(slots=True)
class WeightedBCRecord:
    record: BCStepRecord
    feedback: StepFeedback


@dataclass(slots=True)
class AWBCTrainingSummary:
    initial_loss: float
    final_loss: float
    primitive_accuracy: float
    block_accuracy: float
    dataset_size: int
    epochs: int
    mean_advantage: float
    mean_weight: float


def _partial_overlap_count(
    placements: dict[int, tuple[float, float, float, float]],
) -> int:
    placed = list(placements.values())
    overlap_violations = 0
    for left in range(len(placed)):
        for right in range(left + 1, len(placed)):
            if pairwise_intersection_area(
                torch.tensor(placed[left], dtype=torch.float32),
                torch.tensor(placed[right], dtype=torch.float32),
            ) > 1e-9:
                overlap_violations += 1
    return overlap_violations


def _partial_area_violation_count(
    case: FloorSetCase,
    placements: dict[int, tuple[float, float, float, float]],
) -> int:
    violations = 0
    for block_index, (_x, _y, width, height) in placements.items():
        target_area = float(case.area_targets[block_index].item())
        if target_area <= 0:
            continue
        if abs((width * height) - target_area) / target_area > 0.01:
            violations += 1
    return violations


def _partial_dimension_violation_count(
    case: FloorSetCase,
    placements: dict[int, tuple[float, float, float, float]],
) -> int:
    if case.target_positions is None:
        return 0
    violations = 0
    for block_index, (x, y, width, height) in placements.items():
        fixed = bool(case.constraints[block_index, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
        if not (fixed or preplaced):
            continue
        tx, ty, tw, th = [float(v) for v in case.target_positions[block_index].tolist()]
        if tw > 0 and th > 0 and (abs(width - tw) > 1e-4 or abs(height - th) > 1e-4):
            violations += 1
            continue
        if preplaced and tx >= 0 and ty >= 0 and (abs(x - tx) > 1e-4 or abs(y - ty) > 1e-4):
            violations += 1
    return violations


def _partial_alignment_penalty(
    case: FloorSetCase,
    placements: dict[int, tuple[float, float, float, float]],
) -> float:
    if case.target_positions is None:
        return 0.0
    penalty = 0.0
    for block_index, placement in placements.items():
        target = [float(value) for value in case.target_positions[block_index].tolist()]
        if any(value < 0 for value in target):
            continue
        target_scale = max(float(case.area_targets[block_index].item()), 1.0)
        penalty += (
            sum(
                abs(actual - expected)
                for actual, expected in zip(placement, target, strict=True)
            )
            / target_scale
        )
    return penalty


def _score_state(case: FloorSetCase, state: ExecutionState) -> float:
    placements = state.placements
    if not placements:
        return 0.0

    score = len(placements) / max(case.block_count, 1)
    score -= 0.50 * _partial_overlap_count(placements)
    score -= 0.25 * _partial_area_violation_count(case, placements)
    score -= 0.50 * _partial_dimension_violation_count(case, placements)
    score -= 0.10 * _partial_alignment_penalty(case, placements)

    if len(placements) == case.block_count and case.target_positions is not None:
        ordered_positions = [placements[idx] for idx in range(case.block_count)]
        evaluation = evaluate_positions(case, ordered_positions)
        score += 1.0 if evaluation["official"]["is_feasible"] else -1.0
        score -= min(float(evaluation["official"]["cost"]), 1000.0) / 1000.0

    return float(score)


def _score_candidate_action(
    case: FloorSetCase,
    state: ExecutionState,
    action,
    *,
    candidate_mode: CandidateMode = "semantic",
) -> tuple[float, bool]:
    decision = check_action_mask(case, state, action, mode=candidate_mode)
    if not decision.allowed:
        return -1.0, False

    next_state = ExecutionState(
        placements=dict(state.placements),
        frozen_blocks=set(state.frozen_blocks),
    )
    ActionExecutor(case).apply(next_state, action)
    return _score_state(case, next_state), True


def compute_step_feedback(
    case: FloorSetCase,
    state: ExecutionState,
    action,
    *,
    candidate_mode: CandidateMode = "semantic",
    temperature: float = 1.5,
    min_weight: float = 0.25,
    max_weight: float = 8.0,
) -> StepFeedback:
    verifier_score, positive_legal = _score_candidate_action(
        case,
        state,
        action,
        candidate_mode=candidate_mode,
    )
    negative_actions = generate_negative_actions([action])
    if negative_actions:
        baseline_score, baseline_legal = _score_candidate_action(
            case,
            state,
            negative_actions[0],
            candidate_mode=candidate_mode,
        )
    else:
        baseline_score, baseline_legal = verifier_score - 0.25, None

    advantage = verifier_score - baseline_score
    clipped_advantage = max(min(advantage, 3.0), -3.0)
    weight = math.exp(temperature * clipped_advantage)
    weight = max(min(weight, max_weight), min_weight)
    return StepFeedback(
        verifier_score=float(verifier_score),
        baseline_score=float(baseline_score),
        advantage=float(advantage),
        weight=float(weight),
        positive_legal=positive_legal,
        baseline_legal=baseline_legal,
    )


def build_advantage_dataset_from_cases(
    cases: list[FloorSetCase],
    *,
    max_traces_per_case: int = 2,
    candidate_mode: CandidateMode = "semantic",
) -> list[WeightedBCRecord]:
    dataset: list[WeightedBCRecord] = []
    for case in cases:
        roles = label_case_roles(case)
        traces = generate_pseudo_traces(case, max_traces=max_traces_per_case)
        executor = ActionExecutor(case)
        for trace in traces:
            state = ExecutionState()
            for action in trace.actions:
                feedback = compute_step_feedback(
                    case,
                    state,
                    action,
                    candidate_mode=candidate_mode,
                )
                dataset.append(
                    WeightedBCRecord(
                        record=BCStepRecord(
                            case=case,
                            role_evidence=roles,
                            placements=dict(state.placements),
                            action=action,
                        ),
                        feedback=feedback,
                    )
                )
                executor.apply(state, action)
    return dataset


def _accuracy(
    policy: TypedActionPolicy,
    dataset: list[WeightedBCRecord],
) -> tuple[float, float]:
    primitive_hits = 0
    block_hits = 0
    for item in dataset:
        output = policy(
            item.record.case,
            role_evidence=item.record.role_evidence,
            placements=item.record.placements,
        )
        targets = action_to_targets(item.record.action)
        primitive_target = targets["primitive_id"]
        block_target = targets["block_index"]
        if not isinstance(primitive_target, int) or not isinstance(block_target, int):
            raise TypeError("Typed action targets must contain integer primitive/block ids")
        primitive_hits += int(
            int(output.primitive_logits.argmax().item()) == primitive_target
        )
        block_hits += int(
            int(output.block_logits.argmax().item()) == block_target
        )
    denom = max(len(dataset), 1)
    return primitive_hits / denom, block_hits / denom


def run_advantage_weighted_bc(
    dataset: list[WeightedBCRecord],
    *,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 0,
) -> tuple[TypedActionPolicy, AWBCTrainingSummary]:
    torch.manual_seed(seed)
    policy = TypedActionPolicy(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    initial_loss = None
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for item in dataset:
            optimizer.zero_grad(set_to_none=True)
            breakdown = compute_bc_loss(policy, item.record)
            weighted_loss = breakdown.total * float(item.feedback.weight)
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += float(weighted_loss.item())
        if initial_loss is None:
            initial_loss = epoch_loss / max(len(dataset), 1)

    final_loss = 0.0
    for item in dataset:
        breakdown = compute_bc_loss(policy, item.record)
        final_loss += float((breakdown.total * float(item.feedback.weight)).item())
    final_loss /= max(len(dataset), 1)
    primitive_acc, block_acc = _accuracy(policy, dataset)
    mean_advantage = sum(item.feedback.advantage for item in dataset) / max(len(dataset), 1)
    mean_weight = sum(item.feedback.weight for item in dataset) / max(len(dataset), 1)
    return policy, AWBCTrainingSummary(
        initial_loss=float(initial_loss or 0.0),
        final_loss=float(final_loss),
        primitive_accuracy=float(primitive_acc),
        block_accuracy=float(block_acc),
        dataset_size=len(dataset),
        epochs=epochs,
        mean_advantage=float(mean_advantage),
        mean_weight=float(mean_weight),
    )


def save_policy_checkpoint(
    policy: TypedActionPolicy,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_dim = int(policy.decoder.primitive_head.in_features)
    torch.save(
        {
            "hidden_dim": hidden_dim,
            "state_dict": policy.state_dict(),
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def _adapt_checkpoint_parameter(
    checkpoint_value: torch.Tensor,
    current_value: torch.Tensor,
) -> torch.Tensor | None:
    if checkpoint_value.ndim == 2 and current_value.ndim == 2:
        if (
            checkpoint_value.shape[0] == current_value.shape[0]
            and checkpoint_value.shape[1] + 1 == current_value.shape[1]
        ):
            adapted = torch.zeros_like(current_value)
            adapted[:, : checkpoint_value.shape[1]] = checkpoint_value
            return adapted
    if checkpoint_value.ndim == 1 and current_value.ndim == 1:
        if checkpoint_value.shape[0] + 1 == current_value.shape[0]:
            adapted = torch.zeros_like(current_value)
            adapted[: checkpoint_value.shape[0]] = checkpoint_value
            return adapted
    return None


def load_policy_checkpoint(path: str | Path) -> TypedActionPolicy:
    payload = torch.load(Path(path), map_location="cpu")
    policy = TypedActionPolicy(hidden_dim=int(payload["hidden_dim"]))
    checkpoint_state = payload["state_dict"]
    current_state = policy.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    incompatible = []

    for key, checkpoint_value in checkpoint_state.items():
        if not isinstance(checkpoint_value, torch.Tensor):
            continue
        current_value = current_state.get(key)
        if not isinstance(current_value, torch.Tensor):
            continue
        if checkpoint_value.shape == current_value.shape:
            compatible_state[key] = checkpoint_value
            continue

        adapted_value = _adapt_checkpoint_parameter(
            checkpoint_value,
            current_value,
        )
        if adapted_value is not None:
            compatible_state[key] = adapted_value
            continue
        incompatible.append(key)

    if incompatible:
        warnings.warn(
            "load_policy_checkpoint skipped incompatible tensors: "
            + ", ".join(incompatible),
            UserWarning,
            stacklevel=2,
        )

    policy.load_state_dict(compatible_state, strict=False)
    policy.eval()
    return policy
