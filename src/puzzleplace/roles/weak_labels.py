from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from puzzleplace.data import ConstraintColumns, FloorSetCase


class RoleLabel(StrEnum):
    FIXED_ANCHOR = "fixed_anchor"
    PREPLACED_ANCHOR = "preplaced_anchor"
    BOUNDARY_SEEKER = "boundary_seeker"
    CONNECTIVITY_HUB = "connectivity_hub"
    CLUSTER_MEMBER = "cluster_member"
    MULTI_INSTANCE = "multi_instance"
    AREA_FOLLOWER = "area_follower"


@dataclass(slots=True)
class WeakRoleEvidence:
    block_index: int
    role: RoleLabel
    reasons: list[str]


class WeakRoleLabeler:
    def __init__(self, hub_quantile: float = 0.8):
        self.hub_quantile = hub_quantile

    def label(self, case: FloorSetCase) -> list[WeakRoleEvidence]:
        degrees = self._weighted_degrees(case)
        sorted_degrees = sorted(degrees)
        threshold_index = min(len(sorted_degrees) - 1, int((len(sorted_degrees) - 1) * self.hub_quantile))
        hub_threshold = sorted_degrees[threshold_index] if sorted_degrees else 0.0
        labels: list[WeakRoleEvidence] = []

        for idx in range(case.block_count):
            reasons: list[str] = []
            role = RoleLabel.AREA_FOLLOWER
            constraints = case.constraints[idx]
            if bool(constraints[ConstraintColumns.FIXED].item()):
                role = RoleLabel.FIXED_ANCHOR
                reasons.append("fixed constraint")
            elif bool(constraints[ConstraintColumns.PREPLACED].item()):
                role = RoleLabel.PREPLACED_ANCHOR
                reasons.append("preplaced constraint")
            elif int(constraints[ConstraintColumns.BOUNDARY].item()) != 0:
                role = RoleLabel.BOUNDARY_SEEKER
                reasons.append(f"boundary code={int(constraints[ConstraintColumns.BOUNDARY].item())}")
            elif bool(constraints[ConstraintColumns.MIB].item()):
                role = RoleLabel.MULTI_INSTANCE
                reasons.append("multi-instantiation constraint")
            elif bool(constraints[ConstraintColumns.CLUSTER].item()):
                role = RoleLabel.CLUSTER_MEMBER
                reasons.append("cluster/group constraint")
            elif degrees[idx] >= hub_threshold and degrees[idx] > 0:
                role = RoleLabel.CONNECTIVITY_HUB
                reasons.append(f"weighted degree={degrees[idx]:.3f} >= hub threshold={hub_threshold:.3f}")
            else:
                reasons.append("default weak area follower heuristic")
            labels.append(WeakRoleEvidence(block_index=idx, role=role, reasons=reasons))
        return labels

    def _weighted_degrees(self, case: FloorSetCase) -> list[float]:
        degrees = [0.0 for _ in range(case.block_count)]
        for edge in case.b2b_edges.tolist():
            src, dst, weight = edge
            i, j = int(src), int(dst)
            if i < case.block_count:
                degrees[i] += float(weight)
            if j < case.block_count:
                degrees[j] += float(weight)
        for edge in case.p2b_edges.tolist():
            _pin, block, weight = edge
            j = int(block)
            if j < case.block_count:
                degrees[j] += float(weight)
        return degrees


def label_case_roles(case: FloorSetCase, *, hub_quantile: float = 0.8) -> list[WeakRoleEvidence]:
    return WeakRoleLabeler(hub_quantile=hub_quantile).label(case)
