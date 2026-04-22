from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.repair import finalize_layout
from puzzleplace.repair.overlap_resolver import resolve_overlaps
from puzzleplace.geometry.boxes import pairwise_intersection_area


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="repair-1",
        block_count=3,
        area_targets=torch.tensor([6.0, 6.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        target_positions=torch.tensor(
            [
                [0.0, 0.0, 2.0, 3.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ),
        metrics=None,
    )


def test_finalizer_reduces_overlap_and_returns_ordered_positions() -> None:
    case = _make_case()
    proposed = {
        0: (0.0, 0.0, 2.0, 3.0),
        1: (0.5, 0.0, 2.0, 3.0),
        2: (1.0, 0.0, 2.0, 2.0),
    }
    result = finalize_layout(case, proposed)
    assert len(result.positions) == case.block_count
    assert result.report.total_overlap_area_after <= result.report.total_overlap_area_before
    assert result.report.overlap_pairs_after <= result.report.overlap_pairs_before
    assert result.report.hard_feasible_after is True


def test_resolve_overlaps_breaks_same_row_overlap_chain() -> None:
    positions = {
        12: (2753.0, 0.0, 28.0, 21.0),
        1: (2772.0, 0.0, 28.0, 27.0),
        2: (2782.0, 0.0, 25.0, 23.0),
        16: (2801.0, 0.0, 22.0, 18.0),
        13: (2808.0, 0.0, 25.0, 26.0),
    }
    resolved, _moved = resolve_overlaps(positions)
    boxes = list(resolved.items())
    overlaps = 0
    for left_index, (_left_block, left_box) in enumerate(boxes):
        for _right_block, right_box in boxes[left_index + 1 :]:
            overlaps += int(
                pairwise_intersection_area(
                    torch.tensor(left_box, dtype=torch.float32),
                    torch.tensor(right_box, dtype=torch.float32),
                )
                > 1e-9
            )
    assert overlaps == 0
