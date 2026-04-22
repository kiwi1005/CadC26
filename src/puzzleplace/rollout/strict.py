from __future__ import annotations

from puzzleplace.data import FloorSetCase
from puzzleplace.models.policy import TypedActionPolicy
from puzzleplace.roles import WeakRoleEvidence

from .beam import beam_rollout
from .greedy import RolloutResult, greedy_rollout


def strict_rollout(
    case: FloorSetCase,
    policy: TypedActionPolicy,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    strategy: str = "beam",
) -> RolloutResult:
    if strategy == "greedy":
        return greedy_rollout(case, policy, role_evidence=role_evidence)
    return beam_rollout(case, policy, role_evidence=role_evidence)
