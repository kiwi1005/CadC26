from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import puzzleplace.ml.step7r_swap_replay as replay_module
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.ml.step7r_swap_replay import replay_swap_deck

_FAKE_CASES: dict[int, FloorSetCase] = {}


def _fake_load_validation_cases(
    _base_dir: Path,
    case_ids: list[int],
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
) -> dict[int, FloorSetCase]:
    del floorset_root, auto_download
    return {case_id: _FAKE_CASES[case_id] for case_id in case_ids if case_id in _FAKE_CASES}


def _fake_eval(
    _case: FloorSetCase, positions: list[tuple[float, float, float, float]]
) -> dict[str, Any]:
    x0 = positions[0][0]
    return {
        "quality": {
            "cost": -x0,
            "HPWLgap": -x0,
            "Areagap_bbox": 0.0,
            "Violationsrelative": 0.0,
            "feasible": True,
        }
    }


def _case(case_id: int, positions: list[tuple[float, float, float, float]]) -> FloorSetCase:
    return FloorSetCase(
        case_id=case_id,
        block_count=len(positions),
        area_targets=torch.tensor([w * h for _x, _y, w, h in positions]),
        b2b_edges=torch.empty((0, 3)),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((len(positions), 5)),
        target_positions=torch.tensor(positions),
        metrics=torch.zeros(8),
    )


def _row(case_id: int, pair: tuple[int, int], rank: int) -> dict[str, Any]:
    return {
        "schema": "step7r_swap_source_deck_v1",
        "case_id": str(case_id),
        "deck_rank": rank,
        "parent_example_id": f"ex{rank}",
        "swap_pair": list(pair),
        "post_swap_centers": {},
        "legal": True,
        "rejection_reason": None,
        "source_candidate_id": f"src{rank}",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return path


def _run_replay(
    tmp_path: Path,
    monkeypatch: Any,
    cases: dict[int, FloorSetCase],
    deck_rows: list[dict[str, Any]],
    *,
    n_workers: int = 2,
) -> tuple[dict[str, Any], Path, Path, Path]:
    _FAKE_CASES.clear()
    _FAKE_CASES.update(cases)
    monkeypatch.setattr(replay_module, "load_validation_cases", _fake_load_validation_cases)
    monkeypatch.setattr(replay_module, "official_like_evaluator", _fake_eval)
    examples_path = _write_jsonl(
        tmp_path / "examples.jsonl",
        [{"example_id": row["parent_example_id"]} for row in deck_rows],
    )
    deck_path = _write_jsonl(tmp_path / "deck.jsonl", deck_rows)
    rows_path = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.json"
    failures_path = tmp_path / "failures.json"
    summary = replay_swap_deck(
        tmp_path,
        examples_path,
        deck_path,
        rows_path,
        summary_path,
        failures_path,
        n_workers=n_workers,
    )
    return summary, rows_path, summary_path, failures_path


def test_synthetic_two_block_swap_can_be_strict_winner(tmp_path: Path, monkeypatch: Any) -> None:
    case = _case(1, [(0.0, 0.0, 1.0, 1.0), (10.0, 0.0, 1.0, 1.0)])

    summary, _rows_path, _summary_path, _failures_path = _run_replay(
        tmp_path, monkeypatch, {1: case}, [_row(1, (0, 1), 1)], n_workers=2
    )

    assert summary["strict_meaningful_winner_count"] == 1
    assert summary["phase2_gate_open"] is True
    assert summary["n_workers_used"] == 2


def test_swap_overlap_is_counted_without_strict_winner(tmp_path: Path, monkeypatch: Any) -> None:
    case = _case(
        2,
        [
            (0.0, 0.0, 20.0, 20.0),
            (50.0, 15.0, 10.0, 10.0),
            (62.0, 10.0, 5.0, 20.0),
        ],
    )

    summary, _rows_path, _summary_path, _failures_path = _run_replay(
        tmp_path, monkeypatch, {2: case}, [_row(2, (0, 1), 1)], n_workers=2
    )

    assert summary["strict_meaningful_winner_count"] == 0
    assert summary["overlap_after_splice_count"] >= 1


def test_tiny_fixture_deck_summary_keys_and_types(tmp_path: Path, monkeypatch: Any) -> None:
    case = _case(
        3,
        [(0.0, 0.0, 1.0, 1.0), (10.0, 0.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)],
    )
    deck = [_row(3, (0, 1), 1), _row(3, (0, 2), 2), _row(3, (1, 2), 3)]

    summary, rows_path, summary_path, failures_path = _run_replay(
        tmp_path, monkeypatch, {3: case}, deck, n_workers=2
    )

    assert rows_path.exists()
    assert summary_path.exists()
    assert failures_path.exists()
    assert summary["schema"] == "step7r_swap_replay_summary_v1"
    assert isinstance(summary["request_count"], int)
    assert isinstance(summary["represented_case_count"], int)
    assert isinstance(summary["largest_case_share"], float)
    assert isinstance(summary["fresh_hard_feasible_nonnoop_count"], int)
    assert isinstance(summary["overlap_after_splice_count"], int)
    assert isinstance(summary["soft_regression_rate"], float)
    assert isinstance(summary["bbox_regression_rate"], float)
    assert isinstance(summary["hpwl_regression_rate"], float)
    assert isinstance(summary["actual_all_vector_nonregressing_count"], int)
    assert isinstance(summary["strict_meaningful_winner_count"], int)
    assert isinstance(summary["phase2_gate_open"], bool)
    assert isinstance(summary["n_workers_used"], int)
    assert summary["request_count"] == 3
