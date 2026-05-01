from __future__ import annotations

from puzzleplace.ml.invariant_geometry_decoder import (
    bbox_envelope_shelf_decoder,
    closure_size_bucket,
    order_preserving_slot_decoder,
)


def _payload() -> dict:
    return {
        "candidate_id": "p0",
        "case_id": 0,
        "blocks": [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0, "movable": False},
            {"block_id": 1, "x": 2.0, "y": 0.0, "w": 2.0, "h": 2.0, "movable": True},
            {"block_id": 2, "x": 0.0, "y": 2.0, "w": 2.0, "h": 2.0, "movable": True},
        ],
    }


def test_closure_size_bucket() -> None:
    assert closure_size_bucket(8) == "small_<=10"
    assert closure_size_bucket(15) == "medium_11_20"
    assert closure_size_bucket(30) == "large_21_plus"


def test_bbox_envelope_shelf_decoder_preserves_fixed() -> None:
    decoded = bbox_envelope_shelf_decoder(_payload())
    fixed = next(block for block in decoded["decoded_blocks_preview"] if block["block_id"] == 0)
    assert fixed["x"] == 0.0
    assert fixed["y"] == 0.0
    assert "bbox_envelope" in decoded["invariants"]


def test_order_preserving_slot_decoder_is_non_overlap_for_simple_payload() -> None:
    decoded = order_preserving_slot_decoder(_payload())
    assert decoded["overlap_pair_count"] == 0
    assert "order_topology" in decoded["invariants"]
