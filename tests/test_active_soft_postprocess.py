"""Tests for active-soft boundary repair post-processor."""

from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.repair.active_soft_postprocess import active_soft_postprocess


def _make_case(
    block_count: int = 4,
    boundary_codes: list[int] | None = None,
) -> FloorSetCase:
    """Build a minimal case with boundary constraints."""
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


def test_postprocess_returns_positions_and_report():
    case = _make_case(boundary_codes=[0, 0, 0, 0])
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    result_positions, report = active_soft_postprocess(case, positions)
    assert isinstance(result_positions, list)
    assert isinstance(report, dict)
    assert "active_soft_applied" in report
    assert "active_soft_candidates_evaluated" in report


def test_postprocess_no_boundary_constraints_is_noop():
    case = _make_case(boundary_codes=[0, 0, 0, 0])
    positions = [(0.0, 0.0, 10.0, 10.0)] * 4
    result_positions, report = active_soft_postprocess(case, positions)
    assert result_positions == positions
    assert report["active_soft_applied"] is False
    assert report["active_soft_candidates_evaluated"] == 0
