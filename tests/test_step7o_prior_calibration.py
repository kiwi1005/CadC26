from __future__ import annotations

import json

from puzzleplace.experiments.step7o_prior_calibration import run_prior_calibration


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _candidate(case_id: int, candidate_id: str, *, winner: bool = False) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_candidate_id": candidate_id.rsplit(":", 1)[0],
        "case_id": case_id,
        "decoder": "bbox_envelope_shelf_decoder",
        "target_region": "r1_1",
        "route_class": "regional",
        "closure_size_bucket": "small_<=10",
        "hard_feasible_non_noop": True,
        "non_original_non_noop": True,
        "quality_gate_pass": winner,
        "official_like_improving": winner,
        "metric_regressing": not winner,
        "dominated_by_original": not winner,
        "hpwl_delta": -0.1 if winner else 0.2,
        "bbox_area_delta": 0.0 if winner else 1.0,
        "soft_constraint_delta": 0.0,
        "official_like_cost_delta": -1e-8 if winner else 0.1,
    }


def _archive_candidate(index: int) -> dict[str, object]:
    reasons = ["step7n_g_non_anchor_pareto_front"]
    if index < 3:
        reasons.append("official_like_winner_preservation")
    return {
        "candidate_id": f"n{index}",
        "case_id": 24 if index < 3 else 76,
        "step7n_i_classification": "archive_candidate",
        "step7n_i_selected_for_archive": True,
        "step7n_h_c_retain_reasons": reasons,
    }


def test_prior_calibration_uses_fixed_budget_and_separate_archive_baseline(tmp_path) -> None:
    atlas = tmp_path / "atlas.jsonl"
    _write_jsonl(
        atlas,
        [
            {
                "case_id": 24,
                "feature_family": "training_region_heatmap_prior",
                "region_id": "r1c1",
                "mean_region_prior": 1.0,
                "route_locality_proxy": "regional",
            },
            {
                "case_id": 51,
                "feature_family": "training_region_heatmap_prior",
                "region_id": "r1c1",
                "mean_region_prior": 0.9,
                "route_locality_proxy": "regional",
            },
        ],
    )
    i_candidates = tmp_path / "i.json"
    _write_json(
        i_candidates,
        {
            "rows": [
                {
                    "candidate_id": "case024:a:shelf",
                    "source_candidate_id": "case024:a",
                    "case_id": 24,
                    "target_region": "r1_1",
                    "route_class": "regional",
                    "hard_feasible_non_noop": True,
                    "quality_gate_pass": True,
                    "official_like_improving": True,
                }
            ]
        },
    )
    k_candidates = tmp_path / "k.json"
    rows = [
        _candidate(24, "case024:a:k", winner=True),
        _candidate(24, "case024:b:k", winner=True),
        _candidate(51, "case051:a:k"),
        _candidate(51, "case051:b:k"),
        _candidate(76, "case076:a:k"),
        _candidate(79, "case079:a:k"),
        _candidate(91, "case091:a:k"),
    ]
    _write_json(k_candidates, {"selected_rows": rows})
    n_candidates = tmp_path / "n.json"
    _write_json(
        n_candidates,
        {"rows": [_archive_candidate(i) for i in range(5)]},
    )

    summary = run_prior_calibration(
        atlas,
        i_candidates,
        k_candidates,
        n_candidates,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )
    calibration_rows = [
        json.loads(line)
        for line in (tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected_rows = [row for row in calibration_rows if row["selected_in_top_budget"]]
    selected_case_counts: dict[str, int] = {}
    for row in selected_rows:
        selected_case_counts[str(row["case_id"])] = (
            selected_case_counts.get(str(row["case_id"]), 0) + 1
        )

    assert summary["top_budget_count"] == 6
    assert len(selected_rows) == 6
    assert max(selected_case_counts.values()) == 2
    assert summary["metric_confidence_counts"]["exact_component_comparable"] == 7
    assert summary["metric_confidence_counts"]["exact_component_partial"] == 1
    assert summary["step7ml_winner_baseline"] == 2
    assert summary["step7n_archive_official_like_winner_preservation"]["preserved"] == 3
    assert summary["step7n_non_anchor_pareto_preservation"]["preserved"] == 5
    assert summary["decision"] == "keep_prior_report_only"
    assert summary["concentration_pass"] is False
    assert summary["phase3_gate_open"] is False
    assert summary["gnn_rl_gate_open"] is False
