from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import torch

from puzzleplace.actions import ActionPrimitive, ExecutionState
from puzzleplace.actions.executor import ActionExecutor
from puzzleplace.actions.schema import TypedAction
from puzzleplace.data import FloorSetCase, adapt_validation_batch
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


def _boundary_class(boundary_code: int | None) -> int:
    if boundary_code is None:
        return 0
    mapping = {1: 1, 2: 2, 4: 3, 8: 4}
    return mapping.get(int(boundary_code), 0)


def action_to_targets(action: TypedAction) -> dict[str, int | torch.Tensor | None]:
    primitive_id = list(ActionPrimitive).index(action.primitive)
    geometry = None
    if action.primitive is ActionPrimitive.PLACE_ABSOLUTE and None not in (action.x, action.y, action.w, action.h):
        geometry = torch.tensor([float(action.x), float(action.y), float(action.w), float(action.h)], dtype=torch.float32)
    return {
        "primitive_id": primitive_id,
        "block_index": action.block_index,
        "target_index": action.target_index,
        "boundary_class": _boundary_class(action.boundary_code),
        "geometry": geometry,
    }


def build_bc_dataset_from_cases(cases: list[FloorSetCase], *, max_traces_per_case: int = 2) -> list[BCStepRecord]:
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
