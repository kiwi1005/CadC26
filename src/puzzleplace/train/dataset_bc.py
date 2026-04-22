from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from puzzleplace.actions import (
    ActionPrimitive,
    CandidateMode,
    ExecutionState,
    actions_match,
    generate_candidate_actions,
)
from puzzleplace.actions.executor import ActionExecutor
from puzzleplace.actions.schema import TypedAction
from puzzleplace.data import FloorSetCase, adapt_training_batch, adapt_validation_batch
from puzzleplace.roles import WeakRoleEvidence, label_case_roles
from puzzleplace.trajectory import generate_pseudo_traces

ROOT = Path(__file__).resolve().parents[3]
FLOORSET_ROOT = ROOT / "external" / "FloorSet"
CONTEST_ROOT = FLOORSET_ROOT / "iccad2026contest"
for path in (ROOT, CONTEST_ROOT, FLOORSET_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.download_smoke import _auto_approve_downloads, _ensure_import_paths  # noqa: E402


@dataclass(slots=True)
class BCStepRecord:
    case: FloorSetCase
    role_evidence: list[WeakRoleEvidence]
    placements: dict[int, tuple[float, float, float, float]]
    action: TypedAction


@dataclass(slots=True)
class CandidateRecallSummary:
    total_steps: int
    matched_steps: int
    candidate_mode: CandidateMode

    @property
    def miss_rate(self) -> float:
        return 1.0 - (self.matched_steps / max(self.total_steps, 1))


def _boundary_class(boundary_code: int | None) -> int:
    if boundary_code is None:
        return 0
    mapping = {1: 1, 2: 2, 4: 3, 8: 4}
    return mapping.get(int(boundary_code), 0)


def action_to_targets(action: TypedAction) -> dict[str, int | torch.Tensor | None]:
    primitive_id = list(ActionPrimitive).index(action.primitive)
    geometry = None
    if action.primitive is ActionPrimitive.PLACE_ABSOLUTE and None not in (
        action.x,
        action.y,
        action.w,
        action.h,
    ):
        assert action.x is not None and action.y is not None
        assert action.w is not None and action.h is not None
        geometry = torch.tensor(
            [float(action.x), float(action.y), float(action.w), float(action.h)],
            dtype=torch.float32,
        )
    return {
        "primitive_id": primitive_id,
        "block_index": action.block_index,
        "target_index": action.target_index,
        "boundary_class": _boundary_class(action.boundary_code),
        "geometry": geometry,
    }


def build_bc_dataset_from_cases(
    cases: list[FloorSetCase],
    *,
    max_traces_per_case: int = 2,
) -> list[BCStepRecord]:
    dataset: list[BCStepRecord] = []
    for case in cases:
        roles = label_case_roles(case)
        traces = generate_pseudo_traces(case, max_traces=max_traces_per_case)
        for trace in traces:
            state = ExecutionState()
            executor = ActionExecutor(case)
            for action in trace.actions:
                dataset.append(
                    BCStepRecord(
                        case=case,
                        role_evidence=roles,
                        placements=dict(state.placements),
                        action=action,
                    )
                )
                executor.apply(state, action)
    return dataset


def measure_candidate_recall(
    cases: list[FloorSetCase],
    *,
    max_traces_per_case: int = 2,
    candidate_mode: CandidateMode = "semantic",
) -> CandidateRecallSummary:
    total_steps = 0
    matched_steps = 0
    for case in cases:
        traces = generate_pseudo_traces(case, max_traces=max_traces_per_case)
        for trace in traces:
            state = ExecutionState()
            executor = ActionExecutor(case)
            for action in trace.actions:
                total_steps += 1
                remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
                candidates = generate_candidate_actions(
                    case,
                    state,
                    remaining_blocks=remaining,
                    mode=candidate_mode,
                )
                matched_steps += int(
                    any(
                        actions_match(candidate, action, mode=candidate_mode)
                        for candidate in candidates
                    )
                )
                executor.apply(state, action)
    return CandidateRecallSummary(
        total_steps=total_steps,
        matched_steps=matched_steps,
        candidate_mode=candidate_mode,
    )


def load_validation_cases(*, case_limit: int = 5) -> list[FloorSetCase]:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_validation_dataloader

    dataloader = get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)
    cases: list[FloorSetCase] = []
    for idx, batch in enumerate(dataloader):
        if idx >= case_limit:
            break
        cases.append(adapt_validation_batch(batch, case_id=f"validation-{idx}"))
    return cases


def load_training_cases(
    *,
    case_limit: int = 5,
    batch_size: int = 1,
) -> list[FloorSetCase]:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_training_dataloader

    dataloader = get_training_dataloader(
        data_path=str(FLOORSET_ROOT),
        batch_size=batch_size,
        num_samples=case_limit,
        shuffle=False,
    )
    cases: list[FloorSetCase] = []
    sample_index = 0
    for batch in dataloader:
        batch_tensors = tuple(batch)
        current_batch = int(batch_tensors[0].shape[0])
        for item_index in range(current_batch):
            single = tuple(tensor[item_index : item_index + 1] for tensor in batch_tensors)
            cases.append(adapt_training_batch(single, case_id=f"train-{sample_index}"))
            sample_index += 1
            if sample_index >= case_limit:
                return cases
    return cases
