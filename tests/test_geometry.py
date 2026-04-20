from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.eval.official import evaluate_positions, extract_validation_baseline_metrics
from puzzleplace.geometry import summarize_hard_legality


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="geom-1",
        block_count=2,
        area_targets=torch.tensor([6.0, 6.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 2.0]]),
        target_positions=torch.tensor([[0.0, 0.0, 2.0, 3.0], [3.0, 0.0, 3.0, 2.0]]),
        metrics=torch.tensor([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_summarize_hard_legality_detects_preplaced_and_overlap_violations() -> None:
    case = _make_case()
    valid_positions = [(0.0, 0.0, 2.0, 3.0), (3.0, 0.0, 3.0, 2.0)]
    summary = summarize_hard_legality(case, valid_positions)
    assert summary.is_feasible is True

    invalid_positions = [(0.0, 0.0, 2.0, 3.0), (2.5, 0.0, 3.0, 2.0)]
    invalid_summary = summarize_hard_legality(case, invalid_positions)
    assert invalid_summary.is_feasible is False
    assert invalid_summary.overlap_violations >= 1 or invalid_summary.dimension_violations >= 1


def test_official_wrapper_matches_target_solution_feasibility() -> None:
    case = _make_case()
    positions = [(0.0, 0.0, 2.0, 3.0), (3.0, 0.0, 3.0, 2.0)]
    result = evaluate_positions(case, positions)
    assert result["legality"]["is_feasible"] is True
    assert result["official"]["is_feasible"] is True
    baseline = extract_validation_baseline_metrics(case)
    assert baseline["area_baseline"] == 12.0
