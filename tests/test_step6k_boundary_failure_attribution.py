from __future__ import annotations

import torch

from puzzleplace.data.schema import ConstraintColumns
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.research.boundary_failure_attribution import (
    CandidateEdgeCoverage,
    boundary_role_overlap_audit,
    classify_boundary_failures,
    compact_left_bottom,
    final_bbox_edge_owner_audit,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.research.virtual_frame import PuzzleFrame


def test_final_bbox_edge_owner_audit_detects_regular_edge_stealing() -> None:
    case = make_step6g_synthetic_case(block_count=5)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    placements = {
        0: (1.0, 0.0, 2.0, 2.0),
        1: (4.0, 0.0, 2.0, 2.0),
        2: (1.0, 3.0, 2.0, 2.0),
        3: (-2.0, 0.0, 1.0, 1.0),  # regular block owns xmin
        4: (6.0, 0.0, 1.0, 1.0),
    }

    rows = final_bbox_edge_owner_audit(case, placements)
    left = next(row for row in rows if row["edge"] == "left")

    assert left["owner_block_ids"] == [3]
    assert left["owner_is_regular"] is True
    assert left["regular_or_nonboundary_stole_edge"] is True
    assert left["unsatisfied_required_block_ids"] == [2]


def test_boundary_failure_classification_uses_predicted_hull_mismatch_before_segment_guess() -> (
    None
):
    case = make_step6g_synthetic_case(block_count=5)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[2, ConstraintColumns.BOUNDARY] = 1.0
    pre = {
        0: (1.0, 0.0, 2.0, 2.0),
        1: (4.0, 0.0, 2.0, 2.0),
        2: (0.0, 3.0, 2.0, 2.0),  # satisfied predicted hull left
        3: (1.0, 6.0, 1.0, 1.0),
        4: (6.0, 0.0, 1.0, 1.0),
    }
    post = dict(pre)
    post[3] = (-3.0, 6.0, 1.0, 1.0)  # final bbox expands left after construction
    hull = PuzzleFrame(0.0, 0.0, 8.0, 8.0, density=0.9, variant="unit-hull")
    owners = final_bbox_edge_owner_audit(case, post)

    rows = classify_boundary_failures(
        case,
        pre,
        post,
        predicted_hull=hull,
        edge_owner_rows=owners,
        candidate_coverage=CandidateEdgeCoverage({2: {"left": 3}}),
    )

    assert rows[0]["block_id"] == 2
    assert rows[0]["failure_type"] == "on_predicted_hull_but_not_final_bbox"
    assert "edge_stolen_by_regular_or_nonboundary" in rows[0]["failure_reasons"]


def test_role_overlap_audit_separates_boundary_grouping_and_mib() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    case.constraints[2, ConstraintColumns.CLUSTER] = 7.0
    case.constraints[9, ConstraintColumns.MIB] = 5.0
    placements = {idx: (float(idx * 3), 0.0, 2.0, 2.0) for idx in range(case.block_count)}

    audit = boundary_role_overlap_audit(case, placements)

    assert audit["boundary_plus_grouping_count"] >= 1
    assert audit["boundary_plus_mib_count"] >= 1
    assert audit["unsatisfied_boundary_total"] >= 1


def test_compaction_preserves_hard_legality_and_fixed_anchor() -> None:
    case = make_step6g_synthetic_case(block_count=4)
    case.constraints[:, :] = torch.zeros_like(case.constraints)
    case.constraints[0, ConstraintColumns.FIXED] = 1.0
    case.area_targets = torch.full((4,), 4.0)
    placements = {
        0: (0.0, 0.0, 2.0, 2.0),
        1: (5.0, 0.0, 2.0, 2.0),
        2: (8.0, 0.0, 2.0, 2.0),
        3: (5.0, 4.0, 2.0, 2.0),
    }
    frame = PuzzleFrame(0.0, 0.0, 12.0, 12.0, density=0.9, variant="unit")

    compacted = compact_left_bottom(case, placements, frame=frame, passes=3)

    assert compacted[0] == placements[0]
    assert compacted[1][0] < placements[1][0]
    assert summarize_hard_legality(case, [compacted[idx] for idx in range(4)]).is_feasible
