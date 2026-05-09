from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import replay_causal_requests


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _request(case_id: int, index: int, *, winner: bool) -> dict[str, object]:
    return {
        "schema": "step7p_phase3_causal_request_v1",
        "request_id": f"r{case_id}_{index}",
        "case_id": str(case_id),
        "source_candidate_id": f"cand{case_id}_{index}",
        "intent_family": "soft_guarded_repair",
        "seed_failure_bucket": "soft_regression" if not winner else "bbox_regression",
        "seed_hard_feasible_nonnoop": True,
        "seed_objective_vector": {
            "hpwl_delta": -1.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0 if winner else 1.0,
            "official_like_cost_delta": -1e-6 if winner else 1.0,
        },
    }


def test_replay_causal_requests_reports_strict_gate(tmp_path) -> None:
    requests = [_request(19, i, winner=i < 3) for i in range(80)]
    requests += [_request(24, i, winner=False) for i in range(20)]
    requests_path = _write_jsonl(tmp_path / "requests.jsonl", requests)

    summary = replay_causal_requests(
        requests_path,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
        tmp_path / "failures.json",
    )

    assert summary["request_count"] == 100
    assert summary["strict_meaningful_winner_count"] == 3
    assert summary["non_case024_non_case025_strict_winner_count"] == 3
    assert summary["gnn_rl_gate_open"] is False
