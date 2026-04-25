from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.research.boundary_failure_attribution import final_bbox_edge_owner_audit
from puzzleplace.research.compaction_alternatives import edge_aware_compaction
from puzzleplace.research.hull_stabilization import (
    attribution_cooccurrence,
    hull_drift_metrics,
    hull_stealing_guard_audit,
    select_alternative,
)
from puzzleplace.research.shape_group_probe import shape_group_intervention_probes
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame, final_bbox_boundary_metrics


def test_hull_drift_metrics_reports_regular_owner_drift() -> None:
    case = make_step6g_synthetic_case(block_count=5)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    placements = {
        0: (1.0, 0.0, 2.0, 2.0),
        1: (4.0, 0.0, 2.0, 2.0),
        2: (0.0, 3.0, 2.0, 2.0),
        3: (-3.0, 6.0, 1.0, 1.0),
        4: (6.0, 0.0, 1.0, 1.0),
    }
    predicted = PuzzleFrame(0.0, 0.0, 8.0, 8.0, density=0.9, variant="hull")

    metrics = hull_drift_metrics(case, placements, predicted)

    assert metrics["edge_drift_left"] < 0.0
    assert metrics["drift_owner_is_regular"]["left"] is True
    assert "left" in metrics["drift_summary"]["regular_or_nonboundary_steal_edges"]


def test_attribution_cooccurrence_keeps_overlapping_labels() -> None:
    rows = [
        {
            "failure_type": "on_predicted_hull_but_not_final_bbox",
            "failure_reasons": [
                "on_predicted_hull_but_not_final_bbox",
                "role_conflict_grouping",
            ],
            "unsatisfied_edges": ["left"],
            "required_boundary_type": "left",
            "role_flags": {
                "is_boundary": True,
                "is_grouping": True,
                "is_mib": False,
                "is_terminal_heavy": True,
                "multiple_roles": True,
                "terminal_ratio": 0.7,
            },
        }
    ]

    table = attribution_cooccurrence(rows)

    assert (
        table["highlighted_intersections"][
            "on_predicted_hull_but_not_final_bbox∩grouping"
        ]
        == 1
    )
    assert table["failure_type_x_external_ratio_bucket"][
        "on_predicted_hull_but_not_final_bbox|high"
    ] == 1


def test_hull_stealing_guard_applies_only_without_exceptions() -> None:
    case = make_step6g_synthetic_case(block_count=5)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    placements = {
        0: (1.0, 0.0, 2.0, 2.0),
        1: (4.0, 0.0, 2.0, 2.0),
        2: (1.0, 3.0, 2.0, 2.0),
        3: (-2.0, 6.0, 1.0, 1.0),
        4: (6.0, 0.0, 1.0, 1.0),
    }
    owners = final_bbox_edge_owner_audit(case, placements)
    failures = [
        {
            "unsatisfied_edges": ["left"],
            "candidate_count_for_required_edge": 2,
        }
    ]

    rows = hull_stealing_guard_audit(case, owners, failures)

    assert rows
    assert rows[0]["guard_applied"] is True
    assert rows[0]["boundary_owner_available"] is True


def test_edge_aware_compaction_promotes_boundary_without_breaking_hard_legality() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    case.area_targets = torch.full((4,), 4.0)
    placements = {
        0: (0.0, 0.0, 2.0, 2.0),
        1: (3.0, 0.0, 2.0, 2.0),
        2: (3.0, 3.0, 2.0, 2.0),
        3: (6.0, 0.0, 2.0, 2.0),
    }
    frame = PuzzleFrame(0.0, 0.0, 10.0, 10.0, density=0.9, variant="unit")

    result = edge_aware_compaction(case, placements, frame=frame)

    assert 2 in result.promoted_boundary_block_ids
    assert result.placements[2][0] == 0.0
    assert summarize_hard_legality(case, [result.placements[idx] for idx in range(4)]).is_feasible


def test_shape_probe_can_find_boundary_improving_shape_candidate() -> None:
    case = make_step6g_synthetic_case(block_count=3)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[1, ConstraintColumns.BOUNDARY] = 2.0
    case.area_targets = torch.tensor([4.0, 4.0, 4.0])
    placements = {
        0: (0.0, 0.0, 2.0, 2.0),
        1: (2.5, 3.0, 4.0, 1.0),
        2: (7.0, 0.0, 2.0, 2.0),
    }
    failures = [
        {
            "block_id": 1,
            "failure_type": "shape_mismatch",
            "failure_reasons": ["shape_mismatch"],
        }
    ]
    owners = final_bbox_edge_owner_audit(case, placements)
    frame = PuzzleFrame(0.0, 0.0, 12.0, 8.0, density=0.9, variant="unit")

    probes = shape_group_intervention_probes(case, placements, failures, owners, frame=frame)

    assert probes["summary"]["shape_probe_count"] > 0
    assert any(row["probe_variant"] == "square" for row in probes["shape_probe_records"])
    rate = final_bbox_boundary_metrics(case, placements)["final_bbox_boundary_satisfaction_rate"]
    assert rate < 1.0


def test_select_alternative_prefers_boundary_gain_then_bbox() -> None:
    selected = select_alternative(
        [
            {
                "alternative_type": "original",
                "boundary_satisfaction_rate": 0.2,
                "bbox_area": 100.0,
                "hpwl_proxy": 100.0,
                "hard_feasible": True,
                "frame_max_protrusion_distance": 0.0,
                "frame_num_violations": 0,
            },
            {
                "alternative_type": "edge_aware_compaction",
                "boundary_satisfaction_rate": 0.4,
                "bbox_area": 101.0,
                "hpwl_proxy": 105.0,
                "hard_feasible": True,
                "frame_max_protrusion_distance": 0.0,
                "frame_num_violations": 0,
            },
        ]
    )

    assert selected["selected_alternative_type"] == "edge_aware_compaction"
    assert selected["boundary_gain"] == 0.2
