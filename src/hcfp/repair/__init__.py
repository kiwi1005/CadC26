"""Research-only repair learning sidecar."""

from hcfp.repair.actions import action_sha256, transform_action
from hcfp.repair.replay import repair_replay_dumps, repair_replay_loads
from hcfp.repair.schema import (
    ExpertKind,
    RepairAction,
    RepairCandidate,
    RepairObligation,
    RepairOutcome,
    RepairReplayRecord,
    RepairState,
)
from hcfp.repair.state import build_repair_state


__all__ = [
    "ExpertKind",
    "RepairAction",
    "RepairCandidate",
    "RepairObligation",
    "RepairOutcome",
    "RepairReplayRecord",
    "RepairState",
    "action_sha256",
    "build_repair_state",
    "repair_replay_dumps",
    "repair_replay_loads",
    "transform_action",
]
