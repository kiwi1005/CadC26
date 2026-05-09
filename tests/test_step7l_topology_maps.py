from __future__ import annotations

import torch

from puzzleplace.ml.topology_maps import (
    GridSpec,
    boundary_prior_map,
    build_block_heatmaps,
    target_cell_for_fp_row,
    terminal_weighted_centroid,
    top_cells,
)


def test_terminal_weighted_centroid_uses_matching_block_edges() -> None:
    p2b = torch.tensor([[0.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 99.0]])
    pins = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    centroid = terminal_weighted_centroid(2, p2b, pins)
    assert centroid is not None
    assert centroid[0] == 3.0
    assert centroid[1] == 0.0
    assert centroid[2] == 4.0


def test_boundary_prior_prefers_requested_edge() -> None:
    grid = GridSpec(rows=4, cols=4, x=0.0, y=0.0, w=4.0, h=4.0)
    left = boundary_prior_map(1, grid)
    assert left[0][0] > left[0][3]


def test_build_block_heatmaps_returns_ranked_cells() -> None:
    grid = GridSpec(rows=4, cols=4, x=0.0, y=0.0, w=4.0, h=4.0)
    p2b = torch.tensor([[0.0, 1.0, 1.0]])
    pins = torch.tensor([[3.5, 3.5]])
    maps = build_block_heatmaps(
        block_id=1,
        boundary_code=0,
        p2b_edges=p2b,
        pins_pos=pins,
        grid=grid,
    )
    ranked = top_cells(maps["topology"], k=3)
    assert len(ranked) == 3
    assert ranked[0]["row"] >= 2
    assert ranked[0]["col"] >= 2


def test_target_cell_for_fp_row_uses_center() -> None:
    grid = GridSpec(rows=4, cols=4, x=0.0, y=0.0, w=8.0, h=8.0)
    row, col = target_cell_for_fp_row(torch.tensor([2.0, 2.0, 4.0, 4.0]), grid)
    assert (row, col) == (2, 2)
