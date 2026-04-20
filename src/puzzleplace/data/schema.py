from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch


class ConstraintColumns(IntEnum):
    FIXED = 0
    PREPLACED = 1
    MIB = 2
    CLUSTER = 3
    BOUNDARY = 4


BOUNDARY_CODES = {
    "LEFT": 1,
    "RIGHT": 2,
    "TOP": 4,
    "BOTTOM": 8,
}


@dataclass(slots=True)
class FloorSetCase:
    case_id: str | int
    block_count: int
    area_targets: torch.Tensor
    b2b_edges: torch.Tensor
    p2b_edges: torch.Tensor
    pins_pos: torch.Tensor
    constraints: torch.Tensor
    target_positions: torch.Tensor | None = None
    metrics: torch.Tensor | None = None
    raw: dict[str, Any] | None = None
