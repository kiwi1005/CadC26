"""Shared typed decoder outcome."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from hcfp.repair.schema import RepairAction


class DecodeFailure(str, Enum):
    INVALID_ACTION = "invalid_action"
    IMMOBILE_TARGET = "immobile_target"
    PATCH_BUDGET = "patch_budget"
    HARD_INFEASIBLE = "hard_infeasible"
    NO_DEBT_REDUCTION = "no_debt_reduction"


@dataclass(frozen=True)
class DecodeResult:
    action: RepairAction
    placement: torch.Tensor | None
    failure: DecodeFailure | None
    debt_before: int
    debt_after: int | None
    moved_ids: tuple[int, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.placement is not None and self.failure is None
