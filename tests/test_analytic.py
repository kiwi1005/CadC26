from __future__ import annotations

import torch

from hcfp import analytic
from hcfp.analytic import AnalyticConfig, solve, solve_case, solve_case_with_telemetry
from hcfp.case import from_official
from hcfp.dynamics import DynamicsConfig
from hcfp.geometry import normalize_xywh
from hcfp.runtime import SolveCase
from hcfp.verify import verify_feasible


def test_analytic_solver_returns_verified_geometry_and_exact_hard_targets() -> None:
    case = SolveCase(
        4,
        [4.0, 9.0, 4.0, 4.0],
        [[0, 1, 2.0], [1, 2, 1.0]],
        [],
        [],
        [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[-2.0, 3.0, 2.0, 2.0], [-1.0, -1.0, 3.0, 3.0], [-1.0] * 4, [-1.0] * 4],
    )
    config = AnalyticConfig(DynamicsConfig(population=3, steps=2), projection_iterations=8, direction_beam=2)
    placements = solve(case, config, device="cpu")
    normalized_case = from_official(
        case.block_count,
        case.area_targets,
        case.b2b_connectivity,
        case.p2b_connectivity,
        case.pins_pos,
        case.constraints,
        case.target_positions,
    )

    assert placements[0] == (-2.0, 3.0, 2.0, 2.0)
    assert placements[1][2:] == (3.0, 3.0)
    assert verify_feasible(normalized_case, normalize_xywh(normalized_case, placements))


def test_solve_case_with_telemetry_reports_every_candidate_after_projection(monkeypatch) -> None:
    normalized_case = from_official(
        3,
        [4.0, 4.0, 4.0],
        [[0, 1, 2.0], [1, 2, 1.0]],
        [],
        [],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[-1.0] * 4, [-1.0] * 4, [-1.0] * 4],
    )
    config = AnalyticConfig(DynamicsConfig(population=4, steps=3), projection_iterations=8, direction_beam=2)

    telemetry_calls = 0
    real_telemetry = analytic._telemetry

    def track_telemetry(*args, **kwargs):
        nonlocal telemetry_calls
        telemetry_calls += 1
        return real_telemetry(*args, **kwargs)

    monkeypatch.setattr(analytic, "_telemetry", track_telemetry)
    plain = solve_case(normalized_case, config)
    assert isinstance(plain, torch.Tensor)
    assert telemetry_calls == 0

    result = solve_case_with_telemetry(normalized_case, config)
    telemetry = result.telemetry

    assert telemetry_calls == 1
    assert result.raw_candidates.shape == (9, 3, 4)
    assert result.projected_candidates.shape == (9, 3, 4)
    assert result.energy_history.shape == (4, 3, 3)
    assert result.incumbent_snapshot["safe_source"] == "fallback"
    assert result.incumbent_snapshot["exact_source"] is not None
    assert telemetry.hard_feasible.shape == (9,)
    assert telemetry.raw_overlap.shape == (9,)
    assert telemetry.projected_overlap.shape == (9,)
    assert telemetry.overlap_components.shape == (9,)
    assert telemetry.projection_ok.shape == (9,)
    assert telemetry.projection_active_pairs.shape == (9,)
    assert telemetry.hpwl.shape == (9,)
    assert telemetry.bbox_area.shape == (9,)
    assert telemetry.soft_violation.shape == (9,)
    assert telemetry.projection_displacement.shape == (9,)
    assert len(telemetry.projection_failure_reasons) == 9
    assert bool(telemetry.hard_feasible[0])
    assert torch.all(telemetry.projected_overlap <= telemetry.raw_overlap + 1.0e-5)
    assert torch.all(telemetry.overlap_components >= 0)
    assert torch.all(telemetry.projection_displacement >= 0.0)
    assert verify_feasible(normalized_case, result.selected)
