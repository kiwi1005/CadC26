from __future__ import annotations

import json

import torch

from puzzleplace.alternatives.objective_corridors import (
    generate_corridor_requests,
    request_signature,
)
from puzzleplace.data.schema import FloorSetCase


def _synthetic_case(case_id: int, x_offset: float = 0.0) -> FloorSetCase:
    return FloorSetCase(
        case_id=case_id,
        block_count=4,
        area_targets=torch.tensor([4.0, 4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0], [1.0, 3.0, 1.0]]),
        pins_pos=torch.tensor([[x_offset + 9.0, 1.0], [x_offset + 20.0, 1.0]]),
        constraints=torch.zeros((4, 5)),
        target_positions=torch.tensor(
            [
                [x_offset + 0.0, 0.0, 2.0, 2.0],
                [x_offset + 10.0, 0.0, 2.0, 2.0],
                [x_offset + 20.0, 0.0, 2.0, 2.0],
                [x_offset + 30.0, 0.0, 2.0, 2.0],
            ]
        ),
    )


def test_generate_corridor_requests_schema_summary_and_forbidden_terms() -> None:
    cases = {
        11: _synthetic_case(11),
        12: _synthetic_case(12, x_offset=40.0),
        13: _synthetic_case(13, x_offset=80.0),
    }
    rows, summary = generate_corridor_requests(
        cases,
        grid_size=8,
        candidate_cells_per_block=12,
        max_blocks_per_case=3,
        windows_per_block=3,
        gate_modes=("wire_safe", "bbox_shrink_wire_safe", "soft_repair_budgeted"),
    )

    assert rows
    assert summary["request_count"] == len(rows)
    assert summary["unique_request_signature_count"] == len(
        {request_signature(row) for row in rows}
    )
    assert summary["represented_case_count"] == 3
    assert summary["predicted_hpwl_regression_count_wire_safe"] == 0
    assert summary["predicted_bbox_regression_count_wire_safe"] == 0
    assert summary["predicted_soft_regression_count_wire_safe"] == 0
    assert summary["forbidden_validation_label_terms"] == 0

    first = rows[0]
    assert first["schema"] == "step7m_phase1_objective_corridor_request_v1"
    assert first["request_id"].startswith("step7m_oac_case")
    assert first["accepted_gates"] == {"overlap": True, "hpwl": True, "bbox": True, "soft": True}
    assert first["target_window"]["grid"] == {"rows": 8, "cols": 8}
    assert first["global_report_only"] is False
    assert first["provenance"]["heatmap_used_as"] == "candidate_source_and_tiebreak_only"

    payload = json.dumps({"rows": rows, "summary": summary}, sort_keys=True).lower()
    for forbidden in ("fp_sol", "polygons", "metrics", "target_positions", "oracle_layout"):
        assert forbidden not in payload
