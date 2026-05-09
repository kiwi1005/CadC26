"""Tests for multi-stage active-soft boundary repair post-processor."""

from __future__ import annotations

import torch
import pytest

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.repair.multistage_active_soft import (
    _bbox_of,
    _boundary_edges,
    _boundary_margin,
    _snap_delta,
    _block_center,
    compute_hpwl_sensitivity,
    compute_bbox_edge_owners,
    build_net_neighborhood,
    _rects_overlap,
    _push_to_resolve_overlap,
    multistage_active_soft_postprocess,
)


def _make_case(
    block_count: int = 4,
    boundary_codes: list[int] | None = None,
) -> FloorSetCase:
    if boundary_codes is None:
        boundary_codes = [0] * block_count
    constraints = torch.zeros((block_count, 10), dtype=torch.float32)
    for i, code in enumerate(boundary_codes):
        constraints[i, ConstraintColumns.BOUNDARY] = float(code)
    area = torch.ones(block_count, dtype=torch.float32) * 100.0
    b2b = torch.zeros((0, 3), dtype=torch.float32)
    p2b = torch.zeros((0, 3), dtype=torch.float32)
    pins = torch.zeros((0, 2), dtype=torch.float32)
    targets = torch.zeros((block_count, 4), dtype=torch.float32)
    for i in range(block_count):
        targets[i] = torch.tensor([i * 12.0, 0.0, 10.0, 10.0])
    return FloorSetCase(
        case_id="test",
        block_count=block_count,
        area_targets=area,
        b2b_edges=b2b,
        p2b_edges=p2b,
        pins_pos=pins,
        constraints=constraints,
        target_positions=targets,
        metrics=None,
    )


# --- Unit tests ---


def test_bbox_of():
    positions = [(0.0, 0.0, 10.0, 10.0), (5.0, 3.0, 8.0, 6.0)]
    bbox = _bbox_of(positions)
    assert bbox == (0.0, 0.0, 13.0, 10.0)


def test_boundary_edges():
    assert _boundary_edges(0) == []
    assert _boundary_edges(1) == ["left"]
    assert _boundary_edges(3) == ["left", "right"]
    assert _boundary_edges(15) == ["left", "right", "top", "bottom"]


def test_boundary_margin():
    box = (5.0, 3.0, 10.0, 8.0)
    bbox = (0.0, 0.0, 20.0, 15.0)
    assert _boundary_margin(box, bbox, "left") == 5.0
    assert _boundary_margin(box, bbox, "right") == 5.0  # 20 - (5+10)
    assert _boundary_margin(box, bbox, "top") == 4.0   # 15 - (3+8)
    assert _boundary_margin(box, bbox, "bottom") == 3.0


def test_snap_delta():
    assert _snap_delta("left", 10.0, 1.0) == (-10.0, 0.0)
    assert _snap_delta("right", 10.0, 1.0) == (10.0, 0.0)
    assert _snap_delta("top", 10.0, 1.0) == (0.0, 10.0)
    assert _snap_delta("bottom", 10.0, 1.0) == (0.0, -10.0)
    assert _snap_delta("left", 10.0, 0.5) == (-5.0, 0.0)


def test_block_center():
    assert _block_center((0.0, 0.0, 10.0, 10.0)) == (5.0, 5.0)
    assert _block_center((2.0, 3.0, 4.0, 6.0)) == (4.0, 6.0)


def test_hpwl_sensitivity_no_edges():
    case = _make_case(4, [0, 0, 0, 0])
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    sx, sy = compute_hpwl_sensitivity(case, positions)
    assert sx == [0.0, 0.0, 0.0, 0.0]
    assert sy == [0.0, 0.0, 0.0, 0.0]


def test_bbox_edge_owners():
    positions = [
        (0.0, 0.0, 10.0, 10.0),   # left + bottom
        (10.0, 0.0, 5.0, 10.0),   # right + bottom
        (0.0, 10.0, 15.0, 5.0),   # left + top
    ]
    bbox = (0.0, 0.0, 15.0, 15.0)
    owners = compute_bbox_edge_owners(positions, bbox)
    assert 0 in owners["left"]
    assert 0 in owners["bottom"]
    assert 1 in owners["right"]
    assert 2 in owners["top"]


def test_build_net_neighborhood_empty():
    case = _make_case(4, [0, 0, 0, 0])
    neighbors = build_net_neighborhood(case)
    assert len(neighbors) == 4
    assert neighbors[0] == []


def test_rects_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 10.0, 10.0)
    c = (11.0, 0.0, 10.0, 10.0)
    assert _rects_overlap(a, b)
    assert not _rects_overlap(a, c)


def test_push_to_resolve_overlap():
    moving = (10.0, 5.0, 10.0, 10.0)     # right edge at 20
    obstructing = (18.0, 8.0, 10.0, 8.0)  # left edge at 18, bottom at 8
    push = _push_to_resolve_overlap(moving, obstructing)
    assert push is not None
    pdx, pdy = push
    # Overlap in x: (10+10) - 18 = 2, overlap in y: (5+10) - 8 = 7
    # Smaller push is right (dx=2)
    assert abs(pdx - 2.0) < 1e-9 or abs(pdy - 7.0) < 1e-9


# --- Integration tests ---


def test_multistage_returns_positions_and_report():
    case = _make_case(boundary_codes=[0, 0, 0, 0])
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    result_positions, report = multistage_active_soft_postprocess(case, positions)
    assert isinstance(result_positions, list)
    assert isinstance(report, dict)
    assert "multistage_applied" in report


def test_multistage_no_boundary_constraints_is_noop():
    case = _make_case(boundary_codes=[0, 0, 0, 0])
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    result_positions, report = multistage_active_soft_postprocess(case, positions)
    assert result_positions == positions
    assert report["multistage_applied"] is False
    assert report["multistage_candidates_evaluated"] == 0


def test_multistage_no_target_positions():
    case = _make_case(boundary_codes=[0, 0, 0, 0])
    case.target_positions = None
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    result_positions, report = multistage_active_soft_postprocess(case, positions)
    assert result_positions == positions
    assert report["multistage_applied"] is False
    assert report["multistage_skipped_reason"] == "no_target_positions"
