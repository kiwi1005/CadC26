"""Pure hysteretic latching primitives for constraint activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Side = Literal["left", "right", "top", "bottom"]


@dataclass(frozen=True)
class LatchConfig:
    epsilon_on: float
    epsilon_off: float
    consecutive_on: int = 1

    def __post_init__(self) -> None:
        if not self.epsilon_on < self.epsilon_off:
            raise ValueError("epsilon_on must be smaller than epsilon_off")
        if self.consecutive_on < 1:
            raise ValueError("consecutive_on must be positive")


@dataclass(frozen=True)
class LatchState:
    active: bool = False
    consecutive: int = 0


def update_latch(state: LatchState, gap: float, config: LatchConfig) -> LatchState:
    """Return the next latch state without mutating the previous one."""

    if gap <= config.epsilon_on:
        consecutive = state.consecutive + 1
        return LatchState(
            active=state.active or consecutive >= config.consecutive_on,
            consecutive=consecutive,
        )
    if gap >= config.epsilon_off:
        return LatchState(active=False, consecutive=0)
    return LatchState(active=state.active, consecutive=0)


def side_gap(first: torch.Tensor, second: torch.Tensor, side: Side) -> float:
    """Signed gap from ``first`` side to ``second``.

    Positive means separated, zero means touching, negative means crossing.
    """

    a = _box(first)
    b = _box(second)
    if side == "left":
        return float(a[0] - (b[0] + b[2]))
    if side == "right":
        return float(b[0] - (a[0] + a[2]))
    if side == "top":
        return float(b[1] - (a[1] + a[3]))
    if side == "bottom":
        return float(a[1] - (b[1] + b[3]))
    raise ValueError(f"unknown side: {side}")


def orthogonal_overlap(first: torch.Tensor, second: torch.Tensor, side: Side, *, eps: float = 0.0) -> bool:
    """Whether boxes overlap along the axis orthogonal to a side relation."""

    a = _box(first)
    b = _box(second)
    if side in ("left", "right"):
        overlap = min(float(a[1] + a[3]), float(b[1] + b[3])) - max(float(a[1]), float(b[1]))
    elif side in ("top", "bottom"):
        overlap = min(float(a[0] + a[2]), float(b[0] + b[2])) - max(float(a[0]), float(b[0]))
    else:
        raise ValueError(f"unknown side: {side}")
    return overlap > eps


def side_predicate(
    first: torch.Tensor,
    second: torch.Tensor,
    side: Side,
    *,
    max_gap: float = 0.0,
    overlap_eps: float = 0.0,
) -> bool:
    return side_gap(first, second, side) <= max_gap and orthogonal_overlap(
        first, second, side, eps=overlap_eps
    )


def _box(value: torch.Tensor) -> torch.Tensor:
    box = torch.as_tensor(value, dtype=torch.float64, device="cpu").reshape(-1)
    if box.numel() != 4:
        raise ValueError("box must have four xywh values")
    if not bool(torch.isfinite(box).all()) or bool((box[2:] <= 0).any()):
        raise ValueError("box must be finite with positive width and height")
    return box
