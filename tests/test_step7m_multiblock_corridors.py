from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.experiments.step7m_multiblock_corridors import (
    generate_multiblock_requests,
    make_pair_request,
    replay_multiblock_row,
)
from puzzleplace.geometry.legality import positions_from_case_targets


def _request(block_id: int, x: float) -> dict[str, object]:
    return {
        "case_id": "1",
        "loader_index": 1,
        "request_id": f"r{block_id}",
        "source_family": "micro_axis_corridor",
        "gate_mode": "wire_safe",
        "block_id": block_id,
        "target_window": {"x": x, "y": 0.0, "w": 2.0, "h": 2.0},
        "proxy_objective_vector": {
            "hpwl_delta_proxy": -0.1,
            "bbox_area_delta_proxy": 0.0,
            "boundary_delta_proxy": 0.0,
            "group_delta_proxy": 0.0,
            "mib_delta_proxy": 0.0,
            "overlap_risk_proxy": 0.0,
        },
    }


def _case() -> FloorSetCase:
    return FloorSetCase(
        case_id=1,
        block_count=3,
        area_targets=torch.tensor([4.0, 4.0, 4.0]),
        b2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        p2b_edges=torch.tensor([[-1.0, -1.0, -1.0]]),
        pins_pos=torch.tensor([[-1.0, -1.0]]),
        constraints=torch.zeros((3, 5)),
        target_positions=torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 2.0, 2.0], [8.0, 0.0, 2.0, 2.0]]
        ),
        metrics=torch.zeros(8),
    )


def _eval(
    _case: FloorSetCase, positions: list[tuple[float, float, float, float]], **_kwargs: object
) -> dict[str, object]:
    cost = max(x + w for x, _y, w, _h in positions) - min(x for x, _y, _w, _h in positions)
    return {
        "quality": {
            "cost": cost,
            "HPWLgap": cost,
            "Areagap_bbox": 0.0,
            "Violationsrelative": 0.0,
            "feasible": True,
        }
    }


def test_make_pair_request_requires_same_axis_and_strict_proxy() -> None:
    atlas = {("1", 0): (0.0, 0.0, 2.0, 2.0), ("1", 1): (4.0, 0.0, 2.0, 2.0)}
    pair = make_pair_request(_request(0, 0.5), _request(1, 4.5), atlas)

    assert pair is not None
    assert pair["source_family"] == "paired_micro_axis_strict"
    assert pair["proxy_objective_vector"]["hpwl_delta_proxy"] < 0.0
    assert len(pair["moved_blocks"]) == 2


def test_generate_multiblock_requests_writes_pair_artifacts(tmp_path) -> None:
    requests_path = tmp_path / "requests.jsonl"
    atlas_path = tmp_path / "atlas.jsonl"
    requests_path.write_text(
        "\n".join(
            [__import__("json").dumps(_request(0, 0.5)), __import__("json").dumps(_request(1, 4.5))]
        )
        + "\n"
    )
    atlas_path.write_text(
        "\n".join(
            [
                __import__("json").dumps(
                    {"case_id": "1", "block_id": 0, "current_box": [0.0, 0.0, 2.0, 2.0]}
                ),
                __import__("json").dumps(
                    {"case_id": "1", "block_id": 1, "current_box": [4.0, 0.0, 2.0, 2.0]}
                ),
            ]
        )
        + "\n"
    )
    summary = generate_multiblock_requests(
        requests_path, atlas_path, tmp_path / "out.jsonl", tmp_path / "summary.json"
    )

    assert summary["decision"] == "promote_to_multiblock_replay"
    assert summary["request_count"] == 1
    assert summary["gnn_rl_gate_open"] is False


def test_replay_multiblock_row_records_two_block_exact_move(monkeypatch) -> None:
    import puzzleplace.experiments.step7m_multiblock_corridors as module

    monkeypatch.setattr(module, "evaluate_positions", _eval)
    case = _case()
    baseline = positions_from_case_targets(case)
    before = _eval(case, baseline)
    request = make_pair_request(
        _request(0, 0.5),
        _request(1, 4.5),
        {("1", 0): baseline[0], ("1", 1): baseline[1]},
    )
    assert request is not None
    row = replay_multiblock_row(request, case, baseline, before)  # type: ignore[arg-type]

    assert row["generation_status"] == "realized_exact_multiblock_request"
    assert row["hard_feasible_nonnoop"] is True
    assert row["moved_block_count"] == 2
