from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_request_replay import generate_causal_requests


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(case_id: int, index: int) -> dict[str, object]:
    return {
        "schema": "step7p_phase1_causal_subproblem_row_v1",
        "case_id": str(case_id),
        "subproblem_id": f"sp{case_id}_{index}",
        "seed_candidate_id": f"cand{case_id}_{index}",
        "seed_source": "test",
        "intent_family": ["soft_guarded_repair", "bbox_hull_compaction", "blocker_chain_unblock"][
            index % 3
        ],
        "seed_failure_bucket": ["soft_regression", "bbox_regression", "wrong_slot"][index % 3],
        "metric_confidence": "exact_component_comparable",
        "objective_vector": {
            "hpwl_delta": -1.0,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "official_like_cost_delta": 1.0,
        },
        "hard_feasible_nonnoop": True,
        "affected_block_ids": [index],
        "blocker_block_ids": [],
        "soft_linked_block_ids": [],
        "allowed_repack_families": ["pareto_vector_filter"],
        "validation_label_policy": "labels used for replay/evaluation only, not request generation",
    }


def _unknown_exact_closure_row(case_id: int, index: int, *, forbidden: bool) -> dict[str, object]:
    return {
        "schema": "step7p_phase1_causal_subproblem_row_v1",
        "case_id": str(case_id),
        "subproblem_id": f"sp{case_id}_unknown_{index}",
        "seed_candidate_id": (
            f"case{case_id}_micro_axis_corridor_{index}"
            if forbidden
            else f"case{case_id}_multiblock_closure_{index}"
        ),
        "seed_source": "step7m_phase4",
        "intent_family": "closure_translate_with_repair",
        "seed_failure_bucket": "unknown",
        "metric_confidence": "exact_component_comparable",
        "objective_vector": {
            "hpwl_delta": -0.001,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "official_like_cost_delta": -1e-8,
        },
        "hard_feasible_nonnoop": True,
        "affected_block_ids": [index, index + 100],
        "blocker_block_ids": [],
        "soft_linked_block_ids": [],
        "allowed_repack_families": ["closure_translate_with_repair", "pareto_vector_filter"],
        "validation_label_policy": "labels used for replay/evaluation only, not request generation",
    }


def test_generate_causal_requests_balances_cases_and_forbids_bad_terms(tmp_path) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    atlas = _write_jsonl(
        tmp_path / "atlas.jsonl", [_row(case, i) for case in cases for i in range(12)]
    )
    contract = _write_json(tmp_path / "contract.json", {"phase3_gate_open": True})

    summary = generate_causal_requests(
        atlas,
        contract,
        tmp_path / "requests.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    assert summary["decision"] == "promote_to_bounded_causal_replay"
    assert summary["request_count"] == 96
    assert summary["represented_case_count"] == 8
    assert summary["unique_request_signature_count"] == 96
    assert summary["non_micro_intent_share"] == 1.0
    assert summary["forbidden_request_term_count"] == 0
    assert summary["phase3_replay_gate_open"] is True


def test_generate_causal_requests_uses_non_forbidden_exact_unknown_closure_fallback(
    tmp_path,
) -> None:
    normal_cases = [19, 24, 25, 76, 79, 91, 99]
    rows = [_row(case, i) for case in normal_cases for i in range(12)]
    rows.extend(_unknown_exact_closure_row(51, i, forbidden=False) for i in range(12))
    rows.extend(_unknown_exact_closure_row(51, i + 100, forbidden=True) for i in range(4))
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", rows)
    contract = _write_json(tmp_path / "contract.json", {"phase3_gate_open": True})

    summary = generate_causal_requests(
        atlas,
        contract,
        tmp_path / "requests.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    requests = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert summary["decision"] == "promote_to_bounded_causal_replay"
    assert summary["represented_case_count"] == 8
    assert summary["case_counts"]["51"] == 12
    assert summary["forbidden_request_term_count"] == 0
    assert all("micro_axis_corridor" not in str(row.get("source_candidate_id")) for row in requests)
    assert {
        row["request_source_policy"] for row in requests if row["case_id"] == "51"
    } == {"non_forbidden_exact_closure_coverage_fallback"}
