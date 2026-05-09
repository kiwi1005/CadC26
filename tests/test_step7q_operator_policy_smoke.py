from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_policy_smoke import (
    run_operator_policy_smoke,
    score_example_features,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _example(case_id: int, index: int) -> dict[str, object]:
    intent = "order_preserving_row_repack" if index % 2 == 0 else "closure_translate_with_repair"
    return {
        "schema": "step7q_operator_example_v1",
        "example_id": f"ex_{case_id}_{index:02d}",
        "case_id": str(case_id),
        "source_subproblem_id": f"sp_{case_id}_{index:02d}",
        "source_candidate_id": f"cand_{case_id}_{index:02d}",
        "seed_source": "step7m_phase2",
        "intent_family": intent,
        "seed_failure_bucket": "fb_unknown",
        "metric_confidence": "exact_component_comparable",
        "finite_action_schema_version": "step7q_finite_operator_action_v1",
        "features": {
            "case_id": str(case_id),
            "seed_source": "step7m_phase2",
            "intent_family": intent,
            "seed_failure_bucket": "fb_unknown",
            "metric_confidence": "exact_component_comparable",
            "affected_block_count": 2,
            "blocker_block_count": 0,
            "soft_linked_block_count": 0,
            "moved_block_count": 1,
        },
        "labels": {
            "hard_feasible_nonnoop": True,
            "overlap_after_splice": 0,
            "soft_regression": False,
            "bbox_regression": False,
            "hpwl_regression": False,
            "actual_all_vector_nonregressing": True,
            "strict_meaningful_winner": False,
        },
        "masks": {
            "eligible_for_training": True,
            "forbidden_request_term": False,
        },
    }


def test_score_uses_features_and_masks_not_labels() -> None:
    features = {
        "seed_source": "step7m_phase2",
        "intent_family": "order_preserving_row_repack",
        "seed_failure_bucket": "fb_unknown",
        "metric_confidence": "exact_component_comparable",
        "affected_block_count": 2,
        "blocker_block_count": 0,
        "soft_linked_block_count": 0,
        "moved_block_count": 1,
    }
    good_score, good_components = score_example_features(features, {"eligible_for_training": True})
    bad_score, bad_components = score_example_features(features, {"eligible_for_training": True})

    assert good_score == bad_score
    assert good_components == bad_components
    assert "hard_feasible_nonnoop" not in json.dumps(good_components)
    assert "strict_meaningful_winner" not in json.dumps(good_components)


def test_policy_smoke_selects_label_free_deck_and_keeps_strict_gate_closed(tmp_path) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    examples = [_example(case, index) for case in cases for index in range(15)]
    examples_path = _write_jsonl(tmp_path / "examples.jsonl", examples)
    branch_summary = _write_json(
        tmp_path / "branch.json",
        {
            "best_branch_name": "branch_c_balanced_failure_budget",
            "best_branch_metrics": {
                "request_count": 96,
                "represented_case_count": 8,
                "largest_case_share": 0.25,
                "overlap_after_splice_count": 36,
                "soft_regression_rate": 0.2916666666666667,
                "bbox_regression_rate": 0.0,
                "strict_meaningful_winner_count": 0,
            },
        },
    )

    summary = run_operator_policy_smoke(
        examples_path,
        branch_summary,
        tmp_path / "scores.jsonl",
        tmp_path / "deck.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    deck_text = (tmp_path / "deck.jsonl").read_text(encoding="utf-8")

    assert summary["decision"] == "risk_ranking_smoke_pass_strict_gate_closed"
    assert summary["selected_request_count"] == 96
    assert summary["represented_case_count"] == 8
    assert summary["forbidden_request_term_count"] == 0
    assert summary["score_feature_label_leakage_count"] == 0
    assert summary["risk_profile_pass"] is True
    assert summary["strict_gate_pass"] is False
    assert summary["phase4_gate_open"] is False
    assert "hard_feasible_nonnoop" not in deck_text
    assert "actual_all_vector_nonregressing" not in deck_text
    assert "strict_meaningful_winner" not in deck_text
