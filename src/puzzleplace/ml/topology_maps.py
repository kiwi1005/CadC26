"""Deterministic topology / wire-demand heatmap primitives for Step7L.

These functions are deliberately non-neural baselines. They use only instance
features plus FloorSet training labels when constructing supervised examples;
visible-validation labels are not required or consumed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

EPS = 1e-9
BOUNDARY_LEFT = 1
BOUNDARY_RIGHT = 2
BOUNDARY_TOP = 4
BOUNDARY_BOTTOM = 8


@dataclass(frozen=True, slots=True)
class GridSpec:
    rows: int
    cols: int
    x: float
    y: float
    w: float
    h: float

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.x + (col + 0.5) * self.w / max(self.cols, 1),
            self.y + (row + 0.5) * self.h / max(self.rows, 1),
        )

    def cell_for_point(self, px: float, py: float) -> tuple[int, int]:
        col = min(max(int(((px - self.x) / max(self.w, EPS)) * self.cols), 0), self.cols - 1)
        row = min(max(int(((py - self.y) / max(self.h, EPS)) * self.rows), 0), self.rows - 1)
        return row, col


def valid_block_mask(area: torch.Tensor, fp_sol: torch.Tensor) -> torch.Tensor:
    return (area >= 0) & ~(fp_sol == -1).all(dim=-1)


def frame_from_fp_sol(
    fp_sol: torch.Tensor, mask: torch.Tensor, *, pad_fraction: float = 0.05
) -> GridSpec:
    valid = fp_sol[mask]
    x0 = float(valid[:, 2].min().item())
    y0 = float(valid[:, 3].min().item())
    x1 = float((valid[:, 2] + valid[:, 0]).max().item())
    y1 = float((valid[:, 3] + valid[:, 1]).max().item())
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    pad_x = w * pad_fraction
    pad_y = h * pad_fraction
    return GridSpec(rows=16, cols=16, x=x0 - pad_x, y=y0 - pad_y, w=w + 2 * pad_x, h=h + 2 * pad_y)


def with_grid_size(frame: GridSpec, grid: int) -> GridSpec:
    return GridSpec(rows=grid, cols=grid, x=frame.x, y=frame.y, w=frame.w, h=frame.h)


def normalize_grid(values: list[list[float]]) -> list[list[float]]:
    flat = [max(0.0, float(value)) for row in values for value in row]
    total = sum(flat)
    if total <= EPS:
        denom = max(len(flat), 1)
        return [[1.0 / denom for _ in row] for row in values]
    iterator = iter(flat)
    return [[next(iterator) / total for _ in row] for row in values]


def gaussian_grid(
    grid: GridSpec, center: tuple[float, float], *, sigma: float | None = None
) -> list[list[float]]:
    sigma = sigma or max(grid.w, grid.h) / 5.0
    out: list[list[float]] = []
    for row in range(grid.rows):
        values: list[float] = []
        for col in range(grid.cols):
            cx, cy = grid.cell_center(row, col)
            dist2 = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
            values.append(math.exp(-dist2 / max(2.0 * sigma * sigma, EPS)))
        out.append(values)
    return normalize_grid(out)


def center_prior_map(grid: GridSpec) -> list[list[float]]:
    return gaussian_grid(grid, (grid.x + grid.w / 2.0, grid.y + grid.h / 2.0))


def terminal_weighted_centroid(
    block_id: int,
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
) -> tuple[float, float, float] | None:
    total = 0.0
    sx = 0.0
    sy = 0.0
    for edge in p2b_edges:
        if edge.numel() < 3 or float(edge[0].item()) < 0:
            continue
        pin_id = int(edge[0].item())
        edge_block = int(edge[1].item())
        if edge_block != block_id or pin_id < 0 or pin_id >= int(pins_pos.shape[0]):
            continue
        pin = pins_pos[pin_id]
        if float(pin[0].item()) < 0 or float(pin[1].item()) < 0:
            continue
        weight = max(float(edge[2].item()), 0.0)
        total += weight
        sx += weight * float(pin[0].item())
        sy += weight * float(pin[1].item())
    if total <= EPS:
        return None
    return sx / total, sy / total, total


def terminal_demand_map(
    block_id: int,
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
    grid: GridSpec,
) -> list[list[float]]:
    centroid = terminal_weighted_centroid(block_id, p2b_edges, pins_pos)
    if centroid is None:
        return [[0.0 for _ in range(grid.cols)] for _ in range(grid.rows)]
    return gaussian_grid(grid, (centroid[0], centroid[1]), sigma=max(grid.w, grid.h) / 6.0)


def boundary_prior_map(boundary_code: int, grid: GridSpec) -> list[list[float]]:
    if boundary_code <= 0:
        return [[0.0 for _ in range(grid.cols)] for _ in range(grid.rows)]
    values: list[list[float]] = []
    for row in range(grid.rows):
        row_values: list[float] = []
        for col in range(grid.cols):
            score = 0.0
            if boundary_code & BOUNDARY_LEFT:
                score += 1.0 / (1.0 + col)
            if boundary_code & BOUNDARY_RIGHT:
                score += 1.0 / (1.0 + (grid.cols - 1 - col))
            if boundary_code & BOUNDARY_TOP:
                score += 1.0 / (1.0 + (grid.rows - 1 - row))
            if boundary_code & BOUNDARY_BOTTOM:
                score += 1.0 / (1.0 + row)
            row_values.append(score)
        values.append(row_values)
    return normalize_grid(values)


def blend_maps(weighted_maps: list[tuple[float, list[list[float]]]]) -> list[list[float]]:
    if not weighted_maps:
        raise ValueError("at least one map is required")
    rows = len(weighted_maps[0][1])
    cols = len(weighted_maps[0][1][0]) if rows else 0
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for weight, values in weighted_maps:
        for row in range(rows):
            for col in range(cols):
                out[row][col] += max(weight, 0.0) * max(values[row][col], 0.0)
    return normalize_grid(out)


def target_cell_for_fp_row(fp_row: torch.Tensor, grid: GridSpec) -> tuple[int, int]:
    w, h, x, y = [float(value.item()) for value in fp_row[:4]]
    return grid.cell_for_point(x + w / 2.0, y + h / 2.0)


def top_cells(values: list[list[float]], *, k: int = 16) -> list[dict[str, Any]]:
    cells = [
        {"row": row, "col": col, "score": float(score)}
        for row, scores in enumerate(values)
        for col, score in enumerate(scores)
    ]
    cells.sort(key=lambda cell: (-cell["score"], cell["row"], cell["col"]))
    return cells[:k]


def build_block_heatmaps(
    *,
    block_id: int,
    boundary_code: int,
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
    grid: GridSpec,
) -> dict[str, list[list[float]]]:
    terminal = terminal_demand_map(block_id, p2b_edges, pins_pos, grid)
    boundary = boundary_prior_map(boundary_code, grid)
    center = center_prior_map(grid)
    topology = blend_maps([(0.65, terminal), (0.20, boundary), (0.15, center)])
    return {
        "topology": topology,
        "terminal": normalize_grid(terminal),
        "boundary": normalize_grid(boundary),
        "center": center,
    }
