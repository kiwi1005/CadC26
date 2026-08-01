from __future__ import annotations

from hcfp.analytic import AnalyticConfig, solve
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
