from __future__ import annotations

import json

import torch

from puzzleplace.alternatives.objective_corridors import (
    build_opportunity_atlas,
    summarize_opportunity_atlas,
)
from puzzleplace.data.schema import FloorSetCase


def _synthetic_case() -> FloorSetCase:
    return FloorSetCase(
        case_id=7,
        block_count=4,
        area_targets=torch.tensor([4.0, 4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0], [1.0, 3.0, 1.0]]),
        pins_pos=torch.tensor([[9.0, 1.0], [20.0, 1.0]]),
        constraints=torch.zeros((4, 5)),
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 0.0, 2.0, 2.0],
                [20.0, 0.0, 2.0, 2.0],
                [30.0, 0.0, 2.0, 2.0],
            ]
        ),
    )


def test_opportunity_atlas_schema_and_summary_contract() -> None:
    rows = build_opportunity_atlas({7: _synthetic_case()}, grid_size=8)
    summary = summarize_opportunity_atlas(rows, grid_size=8)

    assert len(rows) == 4
    assert summary["decision"] == "promote_to_objective_corridor_requests"
    assert summary["row_count"] == 4
    assert summary["movable_row_count"] == 4
    assert summary["per_case_counts"]["7"]["total"] == 4
    assert summary["per_case_counts"]["7"]["movable"] == 4
    assert summary["per_case_counts"]["7"]["terminal"] == 2
    assert summary["forbidden_validation_label_terms"] == 0

    first = rows[0]
    assert first["schema"] == "step7m_phase0_block_opportunity_v1"
    assert first["case_id"] == "7"
    assert first["movable"] is True
    assert first["current_wire_proxy"] > 0.0
    assert first["free_slot_count"] > 0
    assert first["provenance"]["source"] == "validation_replay_anchor_geometry"

    payload = json.dumps(rows, sort_keys=True).lower()
    for forbidden in ("fp_sol", "polygons", "metrics", "target_positions", "oracle_layout"):
        assert forbidden not in payload
