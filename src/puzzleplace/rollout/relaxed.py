from __future__ import annotations

from puzzleplace.data import FloorSetCase
from puzzleplace.models.policy import TypedActionPolicy
from puzzleplace.roles import WeakRoleEvidence, label_case_roles

from .semantic import SemanticRolloutResult, semantic_rollout


def relaxed_rollout(
    case: FloorSetCase,
    policy: TypedActionPolicy | None = None,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    max_steps: int | None = None,
) -> SemanticRolloutResult:
    role_evidence = role_evidence or label_case_roles(case)
    result = semantic_rollout(
        case,
        policy,
        role_evidence=role_evidence,
        max_steps=max_steps,
        candidate_mode="relaxed",
    )
    return result
