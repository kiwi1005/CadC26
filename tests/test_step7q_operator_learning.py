from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_learning import (
    build_operator_learning_data_mart,
    finite_action_schema,
    validate_no_label_leakage,
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


def _atlas_row(case_id: int, index: int) -> dict[str, object]:
    return {
        "case_id": str(case_id),
        "subproblem_id": f"sp{case_id}_{index}",
        "seed_candidate_id": f"cand{case_id}_{index}",
        "seed_source": "test",
        "intent_family": "closure_translate_with_repair",
        "seed_failure_bucket": "dominated_by_original",
        "metric_confidence": "exact_component_comparable",
        "route_class": "regional",
        "decoder": "unit",
        "bbox_hull_risk_class": "safe",
        "boundary_constraint_touched": False,
        "mib_constraint_touched": False,
        "group_constraint_touched": True,
        "affected_block_ids": [index],
        "blocker_block_ids": [],
        "soft_linked_block_ids": [],
        "moved_block_count": 1,
        "allowed_repack_families": ["closure_translate_with_repair"],
        "hard_feasible_nonnoop": True,
        "objective_vector": {
            "hpwl_delta": -0.1,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "official_like_cost_delta": 0.1,
        },
        "validation_label_policy": POLICY,
    }


def test_data_mart_separates_features_labels_masks_and_disables_strict_supervision(
    tmp_path,
) -> None:
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    atlas_rows = [_atlas_row(case, index) for case in cases for index in range(41)]
    requests = [
        {
            "request_id": f"req_{index}",
            "source_subproblem_id": row["subproblem_id"],
            "request_source_policy": "direct_causal_attribution",
            "non_micro_intent": True,
        }
        for index, row in enumerate(atlas_rows[:120])
    ]
    replay_rows = [
        {
            "request_id": request["request_id"],
            "fresh_metric_available": True,
            "hard_feasible_nonnoop": True,
            "overlap_after_splice": 0,
            "soft_regression": False,
            "bbox_regression": False,
            "hpwl_regression": False,
            "actual_all_vector_nonregressing": True,
            "strict_meaningful_winner": False,
            "quality_gate_pass": False,
            "actual_objective_vector": {
                "hpwl_delta": -0.1,
                "bbox_area_delta": 0.0,
                "soft_constraint_delta": 0.0,
                "official_like_cost_delta": 0.1,
            },
        }
        for request in requests[:30]
    ]
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", atlas_rows)
    request_path = _write_jsonl(tmp_path / "requests.jsonl", requests)
    replay_path = _write_jsonl(tmp_path / "replay.jsonl", replay_rows)
    blocker = _write_json(
        tmp_path / "blocker.json",
        {"eligible_exact_hard_nonforbidden_count": 120, "strict_meaningful_source_count": 0},
    )
    branch = _write_json(tmp_path / "branch.json", {"best_branch_name": "branch_c"})

    summary = build_operator_learning_data_mart(
        atlas,
        request_path,
        replay_path,
        blocker,
        branch,
        tmp_path / "examples.jsonl",
        tmp_path / "labels.json",
        tmp_path / "features.json",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    examples = [json.loads(line) for line in (tmp_path / "examples.jsonl").read_text().splitlines()]

    assert summary["decision"] == "promote_to_constrained_risk_ranking"
    assert summary["feature_label_leakage_count"] == 0
    assert summary["strict_meaningful_positive_count"] == 0
    assert summary["strict_supervision_enabled"] is False
    assert summary["direct_coordinate_field_count"] == 0
    assert validate_no_label_leakage(examples) == []
    assert "objective_vector" not in json.dumps(examples[0]["features"])
    assert "hard_feasible_nonnoop" not in json.dumps(examples[0]["features"])
    assert "actual_all_vector_nonregressing" in examples[0]["labels"]
    assert examples[0]["masks"]["eligible_for_strict_supervision"] is False
    assert "x" not in finite_action_schema()
    assert "y" not in finite_action_schema()


def test_forbidden_rows_are_masked_not_deleted(tmp_path) -> None:
    row = _atlas_row(19, 0)
    row["seed_candidate_id"] = "bad_micro_axis_corridor"
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", [row])
    empty = _write_jsonl(tmp_path / "empty.jsonl", [])
    blocker = _write_json(
        tmp_path / "blocker.json",
        {"eligible_exact_hard_nonforbidden_count": 0, "strict_meaningful_source_count": 0},
    )
    branch = _write_json(tmp_path / "branch.json", {"best_branch_name": "branch_c"})

    _ = build_operator_learning_data_mart(
        atlas,
        empty,
        empty,
        blocker,
        branch,
        tmp_path / "examples.jsonl",
        tmp_path / "labels.json",
        tmp_path / "features.json",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    example = json.loads((tmp_path / "examples.jsonl").read_text())

    assert example["masks"]["forbidden_request_term"] is True
    assert example["masks"]["eligible_for_training"] is False


def test_forbidden_policy_declaration_does_not_mask_safe_request(tmp_path) -> None:
    row = _atlas_row(19, 0)
    atlas = _write_jsonl(tmp_path / "atlas.jsonl", [row])
    requests = _write_jsonl(
        tmp_path / "requests.jsonl",
        [
            {
                "request_id": "req_safe",
                "source_subproblem_id": row["subproblem_id"],
                "source_candidate_id": row["seed_candidate_id"],
                "seed_source": "test",
                "intent_family": "closure_translate_with_repair",
                "allowed_repack_families": ["closure_translate_with_repair"],
                "request_source_policy": "direct_causal_attribution",
                "forbidden": ["micro_axis_corridor", "soft_repair_budgeted", "hpwl_only"],
                "non_micro_intent": True,
            }
        ],
    )
    empty = _write_jsonl(tmp_path / "empty.jsonl", [])
    blocker = _write_json(
        tmp_path / "blocker.json",
        {"eligible_exact_hard_nonforbidden_count": 1, "strict_meaningful_source_count": 0},
    )
    branch = _write_json(tmp_path / "branch.json", {"best_branch_name": "branch_c"})

    _ = build_operator_learning_data_mart(
        atlas,
        requests,
        empty,
        blocker,
        branch,
        tmp_path / "examples.jsonl",
        tmp_path / "labels.json",
        tmp_path / "features.json",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    example = json.loads((tmp_path / "examples.jsonl").read_text())

    assert example["masks"]["forbidden_request_term"] is False
    assert example["masks"]["eligible_for_training"] is True
    assert example["masks"]["eligible_for_selection"] is True
