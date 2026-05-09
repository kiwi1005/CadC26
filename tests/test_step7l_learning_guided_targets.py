from __future__ import annotations

from pathlib import Path

import torch

from puzzleplace.alternatives.learning_guided_targets import (
    frame_from_visible_inputs,
    requests_from_validation_batch,
    summarize_requests,
)
from puzzleplace.ml.heatmap_dataset import write_jsonl


def _validation_batch() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    area = torch.tensor([[4.0, 9.0, 16.0, -1.0]])
    b2b = torch.tensor([[[-1.0, -1.0, -1.0]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0]]])
    pins = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [10.0, 8.0]]])
    constraints = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        ]
    )
    # Labels are intentionally present to mirror the official loader, but the
    # request generator must ignore them entirely.
    polygons = torch.full((1, 3, 4, 2), 12345.0)
    metrics = torch.full((1, 8), 999.0)
    return (area, b2b, p2b, pins, constraints), (polygons, metrics)


def test_frame_from_visible_inputs_uses_pins_and_area_scale_without_labels() -> None:
    inputs, _labels = _validation_batch()
    frame = frame_from_visible_inputs(inputs[0], inputs[3], grid_size=8)
    assert frame.rows == 8
    assert frame.cols == 8
    assert frame.x < 0.0
    assert frame.y < 0.0
    assert frame.w > 10.0
    assert frame.h > 8.0


def test_requests_from_validation_batch_emits_anchor_and_three_families() -> None:
    rows = [
        row.to_json()
        for row in requests_from_validation_batch(
            _validation_batch(),
            loader_index=24,
            requested_case_id=24,
            grid_size=8,
            top_blocks_per_family=2,
            windows_per_block=2,
        )
    ]
    families = {row["source_family"] for row in rows}
    assert {"original_anchor", "topology", "terminal", "union_diversified"}.issubset(families)
    assert sum(row["is_anchor"] for row in rows) == 1
    assert all(
        row["route_class"] == "unrouted_request_sidecar"
        for row in rows
        if not row["is_anchor"]
    )
    assert all(row["provenance"]["validation_labels_discarded"] for row in rows)
    payload = str(rows).lower()
    assert "polygons" not in payload
    assert "metrics" not in payload
    assert "12345" not in payload
    assert "999" not in payload


def test_summarize_requests_contract_and_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [
        row.to_json()
        for row in requests_from_validation_batch(
            _validation_batch(),
            loader_index=25,
            requested_case_id=25,
            grid_size=8,
            top_blocks_per_family=1,
            windows_per_block=1,
        )
    ]
    path = tmp_path / "requests.jsonl"
    assert write_jsonl(path, rows) == len(rows)
    report = summarize_requests(
        rows,
        request_path=path,
        requested_case_ids=[25],
        grid_size=8,
        top_blocks_per_family=1,
        windows_per_block=1,
    )
    assert report["decision"] == "promote_to_repacker_interface_design_not_replay"
    assert report["uses_validation_target_labels"] is False
    assert report["original_anchor_count"] == 1
    assert report["request_count_by_family"]["topology"] >= 1
