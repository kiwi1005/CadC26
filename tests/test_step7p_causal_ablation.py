from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import run_family_ablation


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_family_ablation_fails_closed_when_no_family_passes(tmp_path) -> None:
    rows = [
        {
            "intent_family": "soft_guarded_repair",
            "case_id": "24",
            "hard_feasible_nonnoop": True,
            "actual_all_vector_nonregressing": False,
            "soft_regression": True,
            "strict_meaningful_winner": False,
        }
        for _ in range(20)
    ]
    replay_rows = _write_jsonl(tmp_path / "replay.jsonl", rows)

    summary = run_family_ablation(
        replay_rows,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    assert summary["decision"] == "complete_step7p_negative_no_family_pass"
    assert summary["passed_family_count"] == 0
    assert summary["gnn_rl_gate_open"] is False


def test_family_ablation_blocks_when_phase3_replay_gate_is_closed(tmp_path) -> None:
    replay_rows = _write_jsonl(tmp_path / "step7p_phase3_causal_replay_rows.jsonl", [])
    _write_json(
        tmp_path / "step7p_phase3_causal_replay_summary.json",
        {
            "schema": "step7p_phase3_causal_replay_summary_v1",
            "decision": "stop_phase3_replay_gate",
            "summary_path": str(tmp_path / "step7p_phase3_causal_replay_summary.json"),
            "phase4_gate_open": False,
        },
    )

    summary = run_family_ablation(
        replay_rows,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    assert summary["decision"] == "blocked_by_phase3_replay_gate"
    assert summary["family_count"] == 0
    assert summary["gnn_rl_gate_open"] is False
