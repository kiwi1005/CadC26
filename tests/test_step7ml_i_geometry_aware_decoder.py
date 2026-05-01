from __future__ import annotations

from puzzleplace.ml.geometry_aware_decoder import (
    generate_slots,
    overlap_pair_count,
    shelf_row_decoder,
    slot_assignment_decoder,
)


def test_generate_slots_stays_inside_bbox() -> None:
    slots = generate_slots({"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}, {"w": 3.0, "h": 2.0})
    assert slots
    assert all(slot["x"] >= 0.0 and slot["y"] >= 0.0 for slot in slots)
    assert all(slot["x"] + slot["w"] <= 10.0 for slot in slots)
    assert all(slot["y"] + slot["h"] <= 10.0 for slot in slots)


def test_shelf_decoder_produces_non_overlap_for_simple_payload() -> None:
    payload = {
        "candidate_id": "c0",
        "case_id": 0,
        "blocks": [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 4.0, "h": 3.0, "movable": True},
            {"block_id": 1, "x": 4.0, "y": 0.0, "w": 3.0, "h": 3.0, "movable": True},
            {"block_id": 2, "x": 0.0, "y": 3.0, "w": 2.0, "h": 2.0, "movable": True},
        ],
    }
    decoded = shelf_row_decoder(payload)
    assert decoded["overlap_pair_count"] == 0
    assert decoded["bbox_containment_failure_count"] == 0


def test_slot_assignment_avoids_fixed_overlap() -> None:
    payload = {
        "candidate_id": "c1",
        "case_id": 0,
        "blocks": [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0, "movable": False},
            {"block_id": 1, "x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0, "movable": True},
            {"block_id": 2, "x": 2.0, "y": 2.0, "w": 2.0, "h": 2.0, "movable": True},
        ],
    }
    decoded = slot_assignment_decoder(payload)
    assert overlap_pair_count(decoded["decoded_blocks_preview"]) == decoded["overlap_pair_count"]
    assert decoded["fixed_preplaced_blocked"] is False
