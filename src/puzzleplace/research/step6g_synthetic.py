from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase


def make_step6g_synthetic_case(case_id: int | str = 0, *, block_count: int = 24) -> FloorSetCase:
    constraints = torch.zeros((block_count, 5), dtype=torch.float32)
    constraints[0, 0] = 1.0  # fixed exact-shape anchor
    constraints[1, 1] = 1.0  # preplaced anchor
    for idx in range(2, block_count, 7):
        constraints[idx, 4] = 1.0
    for idx in range(3, min(block_count, 8)):
        constraints[idx, 3] = 3.0
    for idx in range(8, min(block_count, 12)):
        constraints[idx, 2] = 2.0

    area_targets = torch.tensor(
        [float(4 + (idx % 5)) for idx in range(block_count)], dtype=torch.float32
    )
    target_positions = torch.full((block_count, 4), -1.0, dtype=torch.float32)
    x = 0.0
    y = 0.0
    row_h = 0.0
    for idx in range(block_count):
        w = float(torch.sqrt(area_targets[idx]).item())
        h = float(area_targets[idx].item() / w)
        if x + w > 18.0:
            x = 0.0
            y += row_h + 0.5
            row_h = 0.0
        target_positions[idx] = torch.tensor([x, y, w, h], dtype=torch.float32)
        x += w + 0.5
        row_h = max(row_h, h)

    b2b = []
    for idx in range(block_count - 1):
        b2b.append([float(idx), float(idx + 1), float(1 + (idx % 3))])
    p2b = [[0.0, float(block_count - 1), 2.0], [1.0, 2.0, 1.0]]
    pins = torch.tensor([[0.0, 0.0], [12.0, 8.0]], dtype=torch.float32)
    return FloorSetCase(
        case_id=f"step6g-synth-{case_id}",
        block_count=block_count,
        area_targets=area_targets,
        b2b_edges=torch.tensor(b2b, dtype=torch.float32),
        p2b_edges=torch.tensor(p2b, dtype=torch.float32),
        pins_pos=pins,
        constraints=constraints,
        target_positions=target_positions,
        metrics=None,
    )
