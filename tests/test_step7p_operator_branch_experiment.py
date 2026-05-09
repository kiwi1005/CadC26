from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_operator_branch_experiment import (
    run_operator_branch_experiment,
)

POLICY = "labels used for replay/evaluation only, not request generation"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(case_id: int, index: int, *, hard: bool, soft: float, bbox: float) -> dict[str, object]:
    return {
        "schema": "step7p_phase1_causal_subproblem_row_v1",
        "case_id": str(case_id),
        "subproblem_id": f"case{case_id}_{index}_{hard}_{soft}_{bbox}",
        "seed_candidate_id": f"cand{case_id}_{index}",
        "seed_source": "test",
        "intent_family": "closure_translate_with_repair" if hard else "blocker_chain_unblock",
        "seed_failure_bucket": "dominated_by_original" if hard else "overlap_after_splice",
        "metric_confidence": "exact_component_comparable",
        "objective_vector": {
            "hpwl_delta": -0.1,
            "bbox_area_delta": bbox,
            "soft_constraint_delta": soft,
            "official_like_cost_delta": 0.01,
        },
        "hard_feasible_nonnoop": hard,
        "affected_block_ids": [index],
        "allowed_repack_families": ["pareto_vector_filter"],
        "validation_label_policy": POLICY,
    }


def test_operator_branch_experiment_finds_balanced_partial_branch(tmp_path) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    rows: list[dict[str, object]] = []
    for case in cases:
        for i in range(4):
            rows.append(_row(case, i, hard=True, soft=0.0, bbox=0.0))
        for i in range(4, 12):
            rows.append(_row(case, i, hard=True, soft=1.0, bbox=0.0))
        for i in range(12, 24):
            rows.append(_row(case, i, hard=False, soft=0.0, bbox=0.0))
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", rows)
    baseline = _write_json(
        tmp_path / "baseline.json",
        {
            "request_count": 120,
            "overlap_after_splice_count": 41,
            "soft_regression_rate": 0.525,
            "bbox_regression_rate": 0.15,
            "strict_meaningful_winner_count": 0,
            "phase4_gate_open": False,
        },
    )

    summary = run_operator_branch_experiment(
        atlas,
        baseline,
        tmp_path / "branches.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    assert summary["branch_count"] == 3
    assert summary["effective_partial_branch_count"] >= 1
    assert summary["phase4_open_branch_count"] == 0
    assert summary["gnn_rl_gate_open"] is False
