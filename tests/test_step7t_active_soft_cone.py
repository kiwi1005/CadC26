from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.experiments.step7t_active_soft_cone import (
    active_soft_audit,
    boundary_margins,
    generate_boundary_repair_candidates,
)


def _case() -> FloorSetCase:
    constraints = torch.zeros((2, 5), dtype=torch.float32)
    constraints[0, ConstraintColumns.BOUNDARY] = 4  # top
    return FloorSetCase(
        case_id="synthetic",
        block_count=2,
        area_targets=torch.tensor([4.0, 4.0]),
        b2b_edges=torch.empty((0, 3)),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=constraints,
    )


def test_boundary_margins_detect_top_gap() -> None:
    margins = boundary_margins((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 4.0, 4.0))
    assert margins["top"] == 2.0
    assert margins["left"] == 0.0


def test_generate_boundary_snap_candidate_for_violated_top(monkeypatch) -> None:
    case = _case()
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 2.0, 2.0, 2.0)]

    def fake_eval(_case, _positions, runtime=1.0):
        return {
            "official": {
                "boundary_violations": 1,
                "grouping_violations": 0,
                "mib_violations": 0,
                "total_soft_violations": 1,
                "max_possible_violations": 1,
            }
        }

    monkeypatch.setattr(
        "puzzleplace.experiments.step7t_active_soft_cone.evaluate_positions", fake_eval
    )
    audit = active_soft_audit(case, positions)
    assert audit["active_violated_boundary_components"][0]["violated_edges"] == ["top"]

    candidates = generate_boundary_repair_candidates(
        case,
        positions,
        seed_block=1,
        companion_radius=0.25,
        companion_step=0.25,
    )
    snap = candidates[0]
    assert snap.repair_kind == "boundary_snap_only"
    assert snap.moves == ((0, 0.0, 2.0),)
    assert any(c.repair_kind == "boundary_snap_plus_seed_compensation" for c in candidates)
