from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActionPrimitive(StrEnum):
    PLACE_ABSOLUTE = "place_absolute"
    MOVE = "move"
    RESIZE = "resize"
    PLACE_RELATIVE = "place_relative"
    ALIGN_BOUNDARY = "align_boundary"
    FREEZE = "freeze"


@dataclass(slots=True)
class TypedAction:
    primitive: ActionPrimitive
    block_index: int
    target_index: int | None = None
    boundary_code: int | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    dx: float | None = None
    dy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def require(self, *fields: str) -> None:
        missing = [name for name in fields if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Action {self.primitive} missing required fields: {', '.join(missing)}")
