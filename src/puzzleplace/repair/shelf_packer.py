from __future__ import annotations

from puzzleplace.actions.candidates import _default_dims
from puzzleplace.data import FloorSetCase


def shelf_pack_missing(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
    missing_blocks: list[int],
    *,
    gap: float = 1.0,
) -> tuple[dict[int, tuple[float, float, float, float]], int]:
    packed = dict(positions)
    x_cursor = max((x + width for x, _y, width, _h in packed.values()), default=0.0) + gap
    y_cursor = 0.0
    count = 0
    for block_index in missing_blocks:
        width, height = _default_dims(case, block_index)
        packed[block_index] = (x_cursor, y_cursor, width, height)
        x_cursor += width + gap
        count += 1
    return packed, count
