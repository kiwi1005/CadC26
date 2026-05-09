from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_gnn_rl_readiness import evaluate_gnn_rl_readiness

POLICY = "labels used for replay/evaluation only, not request generation"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_readiness_opens_only_constrained_operator_learning(tmp_path) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    failures = ["soft_regression", "dominated_by_original", "overlap_after_splice", "unknown"]
    intents = [
        "soft_guarded_repair",
        "closure_translate_with_repair",
        "blocker_chain_unblock",
        "hpwl_hull_shrink",
    ]
    rows = [
        {
            "case_id": str(case),
            "seed_failure_bucket": failures[i % len(failures)],
            "intent_family": intents[i % len(intents)],
            "validation_label_policy": POLICY,
        }
        for case in cases
        for i in range(12)
    ]
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", rows)
    branch = _write_json(
        tmp_path / "branch.json",
        {
            "branch_count": 3,
            "effective_partial_branch_count": 1,
            "phase4_open_branch_count": 0,
            "best_branch_name": "branch_c_balanced_failure_budget",
            "best_branch_metrics": {
                "overlap_after_splice_count": 36,
                "soft_regression_rate": 0.29,
                "bbox_regression_rate": 0.0,
            },
        },
    )
    blocker = _write_json(
        tmp_path / "blocker.json",
        {
            "strict_meaningful_source_count": 0,
            "eligible_exact_hard_nonforbidden_count": 107,
            "all_vector_nonregressing_source_count": 30,
        },
    )
    replay = _write_json(
        tmp_path / "replay.json",
        {
            "overlap_after_splice_count": 41,
            "soft_regression_rate": 0.525,
            "bbox_regression_rate": 0.15,
            "phase4_gate_open": False,
        },
    )

    summary = evaluate_gnn_rl_readiness(
        atlas,
        branch,
        blocker,
        replay,
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    assert summary["decision"] == "open_constrained_operator_learning_phase"
    assert summary["gnn_rl_gate_open"] is True
    assert summary["allowed_phase"] == "step7q_constrained_operator_learning"
    assert "contest_runtime_integration" in summary["forbidden_scope"]
