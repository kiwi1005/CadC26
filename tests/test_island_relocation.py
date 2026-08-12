from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hcfp.island_relocation import (
    detect_islands,
    generate_island_relocations,
    relocate_islands,
)
from hcfp.verify import bbox_area, overlap_pairs


def _fragmented() -> torch.Tensor:
    # Blocks 0/1 are the larger core; blocks 2/3 are a rigid, internally
    # touching island separated by a large empty region.
    return torch.tensor(
        (
            (0.0, 0.0, 2.0, 2.0),
            (2.0, 0.0, 2.0, 2.0),
            (10.0, 0.0, 1.0, 1.0),
            (10.0, 1.0, 1.0, 1.0),
        )
    )


def test_detects_contact_and_proximity_islands_and_accepts_explicit_components() -> (
    None
):
    boxes = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (1.01, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 1.0),
        )
    )

    assert detect_islands(boxes) == ((0,), (1,), (2,))
    assert detect_islands(boxes, proximity=0.02) == ((0, 1), (2,))
    generated = generate_island_relocations(
        boxes,
        components=((0, 1), (2,)),
        max_candidates=8,
    )
    assert generated
    assert all(candidate.members == (2,) for candidate in generated)


def test_relocates_fragmented_island_without_overlap_or_internal_geometry_change() -> (
    None
):
    boxes = _fragmented()
    result = relocate_islands(boxes, max_candidates=8)

    assert result.moved
    assert result.core == (0, 1)
    assert result.islands == ((0, 1), (2, 3))
    assert bbox_area(result.placement) < bbox_area(boxes)
    assert not overlap_pairs(result.placement)

    # Rigid relocation changes neither shape/area nor pairwise geometry inside
    # the moved island.
    assert torch.equal(result.placement[2:, 2:4], boxes[2:, 2:4])
    assert torch.equal(
        result.placement[2:, 2] * result.placement[2:, 3], boxes[2:, 2] * boxes[2:, 3]
    )
    old_offsets = boxes[3, :2] - boxes[2, :2]
    new_offsets = result.placement[3, :2] - result.placement[2, :2]
    assert torch.equal(new_offsets, old_offsets)
    assert torch.equal(result.placement[0:2], boxes[0:2])


def test_preplaced_members_are_never_moved_and_pin_median_is_bounded() -> None:
    boxes = _fragmented()
    protected = SimpleNamespace(
        preplaced_mask=torch.tensor([False, False, True, False]),
        p2b_edges=torch.tensor(((0.0, 2.0, 1.0), (1.0, 3.0, 3.0))),
        pins=torch.tensor(((5.0, 5.0), (7.0, 7.0))),
        b2b_weight=torch.zeros((4, 4)),
    )

    result = relocate_islands(protected, boxes)
    assert not result.moved
    assert torch.equal(result.placement, boxes)
    assert result.candidates == ()

    movable = SimpleNamespace(
        preplaced_mask=torch.zeros(4, dtype=torch.bool),
        p2b_edges=protected.p2b_edges,
        pins=protected.pins,
        b2b_weight=protected.b2b_weight,
    )
    candidates = generate_island_relocations(movable, boxes, max_candidates=16)
    pin = next(
        candidate
        for candidate in candidates
        if candidate.strategy == "pin_weighted_median"
    )
    # Weighted median chooses pin (7, 7), while the weighted median source
    # center is (10.5, 1.5), yielding this rigid translation.
    assert pin.delta == pytest.approx((-3.5, 5.5))
    assert not overlap_pairs(pin.placement)
