from __future__ import annotations

import torch

from puzzleplace.data import FloorSetCase
from puzzleplace.repair import RepairReport
from puzzleplace.scoring import ObjectiveCandidate, select_objective_candidate


def _case() -> FloorSetCase:
    return FloorSetCase(
        case_id="selector-test",
        block_count=2,
        area_targets=torch.tensor([4.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((2, 5)),
    )


def _report(moved: int = 0) -> RepairReport:
    return RepairReport(
        hard_feasible_before=True,
        hard_feasible_after=True,
        overlap_pairs_before=0,
        overlap_pairs_after=0,
        total_overlap_area_before=0.0,
        total_overlap_area_after=0.0,
        area_violations_before=0,
        area_violations_after=0,
        dimension_violations_before=0,
        dimension_violations_after=0,
        mean_displacement=float(moved),
        max_displacement=float(moved),
        preserved_x_order_fraction=1.0,
        shelf_fallback_count=0,
        moved_block_count=moved,
    )


def test_objective_proxy_selects_compact_connected_candidate() -> None:
    case = _case()
    spread = ObjectiveCandidate(
        source_id="spread",
        positions=[(0.0, 0.0, 2.0, 2.0), (100.0, 0.0, 2.0, 2.0)],
        repair_report=_report(),
        semantic_placed_fraction=1.0,
        semantic_fallback_fraction=0.0,
    )
    compact = ObjectiveCandidate(
        source_id="compact",
        positions=[(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)],
        repair_report=_report(),
        semantic_placed_fraction=1.0,
        semantic_fallback_fraction=0.0,
    )

    selection = select_objective_candidate(case, [spread, compact])

    assert selection.candidate.source_id == "compact"
    assert selection.ranked_indices == [1, 0]
