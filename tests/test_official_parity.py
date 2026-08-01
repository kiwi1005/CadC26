from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hcfp.case import from_official
from hcfp.geometry import normalize_xywh
from hcfp.verify import (
    bbox_area,
    compute_cost,
    compute_total_score,
    exact_metrics,
    soft_violation_normalized,
    total_hpwl,
    verify,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "artifacts" / "floorset-v10"
EVALUATOR = REFERENCE_ROOT / "iccad2026contest" / "iccad2026_evaluate.py"


def _official():
    if not EVALUATOR.is_file():
        pytest.skip("pinned FloorSet reference checkout is not present")
    pytest.importorskip("shapely")
    sys.path.insert(0, str(REFERENCE_ROOT))
    spec = importlib.util.spec_from_file_location("_hcfp_official_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator from {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hard_predicates_match_pinned_official_v10() -> None:
    official = _official()
    positions = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0 - 2.0e-6, 2.0 - 2.0e-6, 2.0, 2.0),
        (5.0, 0.0, 2.001, 3.0),
        (8.0, 0.0, 1.0, 1.0),
    ]
    areas = torch.tensor([4.0, 4.0, 6.0, 1.0])
    constraints = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    )
    targets = [
        (-1.0, -1.0, -1.0, -1.0),
        (-1.0, -1.0, -1.0, -1.0),
        (-1.0, -1.0, 2.0, 3.0),
        (8.001, 0.0, 1.0, 1.0),
    ]
    raw_case = {
        "area_targets": areas,
        "constraints": constraints,
        "target_positions": targets,
    }
    ours = verify(raw_case, positions)

    assert len(ours.overlap_pairs) == official.check_overlap(positions)
    assert len(ours.area_bad) == official.check_area_tolerance(positions, areas, skip_indices={2, 3})
    assert len(ours.fixed_bad) + len(ours.preplaced_bad) == official.check_dimension_hard_constraints(
        positions, targets, constraints, len(positions)
    )


def test_soft_metrics_hpwl_and_bbox_match_pinned_official_v10() -> None:
    official = _official()
    positions = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 2.0, 2.0),
        (0.0, 3.0, 1.0, 4.0),
        (4.0, 3.0, 1.0, 4.0),
    ]
    areas = torch.tensor([4.0, 4.0, 4.0, 4.0])
    b2b = torch.tensor([[0.0, 3.0, 2.0], [1.0, 3.0, 1.0], [-1.0, -1.0, -1.0]])
    p2b = torch.tensor([[0.0, 2.0, 0.5], [-1.0, -1.0, -1.0]])
    pins = torch.tensor([[0.0, 7.0], [-1.0, -1.0]])
    constraints = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 8],
            [0, 0, 2, 2, 4],
            [0, 0, 2, 2, 2],
        ]
    )
    baseline = {
        "hpwl_baseline": official.calculate_hpwl_b2b(positions, b2b)
        + official.calculate_hpwl_p2b(positions, p2b, pins),
        "area_baseline": official.calculate_bbox_area(positions),
    }
    official_metrics = official.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        baseline,
        constraints,
        b2b,
        p2b,
        pins,
        areas,
        median_runtime=1.0,
    )

    case = from_official(4, areas, b2b, p2b, pins, constraints)
    normalized = normalize_xywh(case, positions)
    ours = soft_violation_normalized(case, normalized)

    assert verify(case, normalized).feasible == official_metrics.is_feasible
    assert total_hpwl(case, normalized) == pytest.approx(official_metrics.hpwl_total, abs=1.0e-5)
    assert bbox_area(positions) == pytest.approx(official_metrics.bbox_area)
    assert ours.raw_boundary == official_metrics.boundary_violations
    assert ours.raw_grouping == official_metrics.grouping_violations
    assert ours.raw_mib == official_metrics.mib_violations
    assert ours.maximum == official_metrics.max_possible_violations
    assert ours.total == pytest.approx(official_metrics.violations_relative)
    exact = exact_metrics(
        case,
        normalized,
        baseline_hpwl=baseline["hpwl_baseline"],
        baseline_area=baseline["area_baseline"],
    )
    assert exact.cost == pytest.approx(official_metrics.cost)


def test_cost_and_weighted_total_match_pinned_official_v10() -> None:
    official = _official()
    inputs = [
        (-0.2, 0.3, 0.0, 0.5, True),
        (0.4, -0.1, 0.25, 1.7, True),
        (0.0, 0.0, 0.0, 1.0, False),
    ]
    for hpwl_gap, area_gap, soft, runtime_factor, feasible in inputs:
        assert compute_cost(hpwl_gap, area_gap, soft, runtime_factor, feasible) == pytest.approx(
            official.compute_cost(hpwl_gap, area_gap, soft, runtime_factor, feasible)
        )

    costs = [1.0, 2.0, 3.0, 4.0]
    counts = [21, 64, 96, 120]
    assert compute_total_score(costs, counts) == pytest.approx(official.compute_total_score(costs, counts))
