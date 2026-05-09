from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_causal_subproblem_atlas import build_causal_subproblem_atlas


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _candidate(case_id: int, index: int) -> dict[str, object]:
    failure_mode = index % 5
    row: dict[str, object] = {
        "candidate_id": f"case{case_id}:cand{index}",
        "case_id": case_id,
        "hard_feasible_non_noop": True,
        "moved_block_count": 3,
        "decoded_blocks_preview": [
            {"block_id": index, "boundary": int(index % 7 == 0), "mib": int(index % 11 == 0)},
            {"block_id": index + 100, "boundary": 0, "mib": 0},
        ],
        "closure_bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
        "official_like_cost_delta": 0.1,
        "hpwl_delta": 0.0,
        "bbox_area_delta": 0.0,
        "soft_constraint_delta": 0.0,
    }
    if failure_mode == 0:
        row["soft_constraint_delta"] = 1.0
    elif failure_mode == 1:
        row["bbox_area_delta"] = 1.0
    elif failure_mode == 2:
        row["hpwl_delta"] = -1.0
        row["official_like_cost_delta"] = 1.0
    elif failure_mode == 3:
        row["no_slot_available"] = True
    else:
        row["dominated_by_original"] = True
    return row


def test_causal_subproblem_atlas_builds_gateable_causal_rows(tmp_path) -> None:
    lock = _write_json(
        tmp_path / "lock.json",
        {"decision": "start_causal_closure_repack", "phase1_gate_open": True},
    )
    cases = [19, 24, 25, 51, 76, 79, 91, 99]
    i_rows = [_candidate(case_id, index) for case_id in cases for index in range(10)]
    i_candidates = _write_json(tmp_path / "i.json", {"rows": i_rows})
    k_candidates = _write_json(tmp_path / "k.json", {"selected_rows": []})
    phase2_rows = _write_jsonl(tmp_path / "m2_rows.jsonl", [])
    phase4_rows = _write_jsonl(tmp_path / "m4_rows.jsonl", [])
    phase2_summary = _write_json(tmp_path / "m2.json", {"replay_rows_path": str(phase2_rows)})
    phase4_summary = _write_json(tmp_path / "m4.json", {"replay_rows_path": str(phase4_rows)})
    target_quality = _write_json(
        tmp_path / "target.json",
        {"target_failure_bucket_counts_retained": {"bad_internal_repack": 4}},
    )

    summary = build_causal_subproblem_atlas(
        lock,
        i_candidates,
        k_candidates,
        phase2_summary,
        phase4_summary,
        target_quality,
        cases,
        tmp_path / "atlas.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
        tmp_path / "failures.json",
    )

    assert summary["decision"] == "promote_to_synthetic_causal_repacker"
    assert summary["subproblem_count"] == 80
    assert summary["represented_case_count"] == 8
    assert summary["largest_case_share"] == 0.125
    assert summary["nonzero_intent_family_count"] >= 4
    assert summary["forbidden_validation_label_term_count"] == 0
    first_row = json.loads((tmp_path / "atlas.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_row["schema"] == "step7p_phase1_causal_subproblem_row_v1"
    assert first_row["validation_label_policy"]
