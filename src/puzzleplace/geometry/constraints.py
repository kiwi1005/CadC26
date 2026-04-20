from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HardLegalitySummary:
    is_feasible: bool
    overlap_violations: int
    area_violations: int
    dimension_violations: int
