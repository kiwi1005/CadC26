"""Research-only repair learning sidecar."""

from hcfp.repair.actions import action_sha256, transform_action
from hcfp.repair.replay import (
    repair_generation_dumps,
    repair_generation_loads,
    repair_replay_dumps,
    repair_replay_loads,
)
from hcfp.repair.model import ContactRepairModel, RepairModelConfig
from hcfp.repair.schema import (
    ExpertKind,
    RepairAction,
    RepairCandidate,
    RepairGenerationRecord,
    RepairObligation,
    RepairOutcome,
    RepairReplayRecord,
    RepairState,
)
from hcfp.repair.state import build_repair_state


__all__ = [
    "ExpertKind",
    "ContactRepairModel",
    "RepairAction",
    "RepairCandidate",
    "RepairGenerationRecord",
    "RepairObligation",
    "RepairOutcome",
    "RepairReplayRecord",
    "RepairModelConfig",
    "RepairState",
    "action_sha256",
    "build_repair_state",
    "repair_replay_dumps",
    "repair_replay_loads",
    "repair_generation_dumps",
    "repair_generation_loads",
    "transform_action",
]
