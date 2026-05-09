from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_parameter_expansion import (
    action_variants,
    run_operator_parameter_expansion,
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
    family = "order_preserving_row_repack" if index % 2 == 0 else "closure_translate_with_repair"
    return {
        "example_id": f"ex_{case_id}_{index:02d}",
        "case_id": str(case_id),
        "source_subproblem_id": f"sp_{case_id}_{index:02d}",
        "source_candidate_id": f"cand_{case_id}_{index:02d}",
        "features": {
            "case_id": str(case_id),
            "intent_family": family,
            "seed_failure_bucket": "fb_unknown",
            "affected_block_count": 3,
            "blocker_block_count": 0,
            "soft_linked_block_count": 0,
            "moved_block_count": 1,
        },
        "labels": {
            "official_like_cost_delta": -1.0e-8,
            "strict_meaningful_winner": False,
        },
    }


def _deck_row(case_id: int, index: int, rank: int) -> dict[str, object]:
    return {
        "deck_rank": rank,
        "example_id": f"ex_{case_id}_{index:02d}",
        "case_id": str(case_id),
        "source_subproblem_id": f"sp_{case_id}_{index:02d}",
        "source_candidate_id": f"cand_{case_id}_{index:02d}",
        "policy_score": 5.0 + index / 100.0,
    }


def test_action_variants_are_finite_and_coordinate_free() -> None:
    variants = action_variants(
        {
            "intent_family": "closure_translate_with_repair",
            "seed_failure_bucket": "fb_unknown",
            "affected_block_count": 3,
            "blocker_block_count": 0,
            "soft_linked_block_count": 0,
        }
    )

    assert variants
    assert all(action["vector_guard_mode"] == "all_vector_nonregress" for action in variants)
    assert all(action["direct_coordinate_fields"] == [] for action in variants)
    assert all("x" not in action for action in variants)
    assert all("y" not in action for action in variants)


def test_parameter_expansion_prepares_replay_deck_without_strict_claim(tmp_path) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    examples = [_example(case, index) for case in cases for index in range(12)]
    deck = [
        _deck_row(case, index, len(cases) * index + offset + 1)
        for index in range(12)
        for offset, case in enumerate(cases)
    ]
    examples_path = _write_jsonl(tmp_path / "examples.jsonl", examples)
    deck_path = _write_jsonl(tmp_path / "deck.jsonl", deck)
    policy_summary = _write_json(
        tmp_path / "policy.json",
        {"decision": "risk_ranking_smoke_pass_strict_gate_closed", "risk_profile_pass": True},
    )

    summary = run_operator_parameter_expansion(
        examples_path,
        deck_path,
        policy_summary,
        tmp_path / "candidates.jsonl",
        tmp_path / "expansion_deck.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    deck_text = (tmp_path / "expansion_deck.jsonl").read_text(encoding="utf-8")

    assert summary["decision"] == "parameter_expansion_deck_ready_for_fresh_replay"
    assert summary["selected_expansion_count"] == 96
    assert summary["represented_case_count"] == 8
    assert summary["forbidden_action_term_count"] == 0
    assert summary["direct_coordinate_field_count"] == 0
    assert summary["fresh_replay_required"] is True
    assert summary["strict_winner_evidence_count"] == 0
    assert summary["phase4_gate_open"] is False
    assert "hard_feasible_nonnoop" not in deck_text
    assert "strict_meaningful_winner" not in deck_text
