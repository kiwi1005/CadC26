from __future__ import annotations

import torch

from puzzleplace.geometry.boxes import pairwise_intersection_area


def _overlap_stats(
    block_index: int,
    candidate: tuple[float, float, float, float],
    positions: dict[int, tuple[float, float, float, float]],
) -> tuple[int, float]:
    overlap_count = 0
    overlap_area = 0.0
    candidate_tensor = torch.tensor(candidate, dtype=torch.float32)
    for other_block, other_box in positions.items():
        if other_block == block_index:
            continue
        area = pairwise_intersection_area(
            candidate_tensor,
            torch.tensor(other_box, dtype=torch.float32),
        )
        if area > 1e-9:
            overlap_count += 1
            overlap_area += area
    return overlap_count, overlap_area


def resolve_overlaps(
    positions: dict[int, tuple[float, float, float, float]],
    *,
    locked_blocks: set[int] | None = None,
    gap: float = 1.0,
) -> tuple[dict[int, tuple[float, float, float, float]], set[int]]:
    locked_blocks = locked_blocks or set()
    resolved = dict(positions)
    moved: set[int] = set()
    order = sorted(
        resolved,
        key=lambda idx: (
            idx not in locked_blocks,
            resolved[idx][1],
            resolved[idx][0],
        ),
    )
    for _iteration in range(len(order) * 3):
        changed_any = False
        for block_index in order:
            if block_index in locked_blocks:
                continue
            x, y, width, height = resolved[block_index]
            blockers = []
            for other_block, other_box in resolved.items():
                if other_block == block_index:
                    continue
                overlap_area = pairwise_intersection_area(
                    torch.tensor((x, y, width, height), dtype=torch.float32),
                    torch.tensor(other_box, dtype=torch.float32),
                )
                if overlap_area > 1e-9:
                    blockers.append(other_box)
            if not blockers:
                continue
            right_shift = (
                max(ox + ow + gap for ox, _oy, ow, _oh in blockers),
                y,
                width,
                height,
            )
            down_shift = (
                x,
                max(oy + oh + gap for _ox, oy, _ow, oh in blockers),
                width,
                height,
            )
            candidates = []
            for candidate in (right_shift, down_shift):
                count, area = _overlap_stats(block_index, candidate, resolved)
                displacement = abs(candidate[0] - x) + abs(candidate[1] - y)
                candidates.append((count, area, displacement, candidate))
            _count, _area, _displacement, chosen = min(candidates, key=lambda item: item[:3])
            resolved[block_index] = chosen
            moved.add(block_index)
            changed_any = True
        if not changed_any:
            break
    return resolved, moved
