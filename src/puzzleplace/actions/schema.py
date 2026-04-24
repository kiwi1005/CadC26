from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionPrimitive(str, Enum):
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
            raise ValueError(
                f"Action {self.primitive} missing required fields: {', '.join(missing)}"
            )


def canonical_action_key(action: TypedAction) -> tuple[object, ...]:
    """Return a stable, metadata-free key for comparing/logging actions.

    Step6 research diagnostics compare candidate choices across rollout
    continuations.  Metadata may contain heuristic/debug fields and must not
    affect those comparisons, so the canonical key is limited to the typed
    action contract itself.
    """

    return (
        action.primitive.value,
        int(action.block_index),
        None if action.target_index is None else int(action.target_index),
        None if action.boundary_code is None else int(action.boundary_code),
        action.x,
        action.y,
        action.w,
        action.h,
        action.dx,
        action.dy,
    )
