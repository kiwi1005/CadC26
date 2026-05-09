from __future__ import annotations

import json

from puzzleplace.ml.training_demand_prior import (
    build_input_inventory,
    build_training_demand_atlas,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_input_inventory_reads_stop_guard_and_label_separation(tmp_path) -> None:
    step7n = tmp_path / "step7n.json"
    inventory = tmp_path / "inventory.json"
    schema = tmp_path / "schema.json"
    quality = tmp_path / "quality.json"
    _write(
        step7n,
        {
            "decision": "stop_no_archive_signal",
            "strict_archive_candidate_count": 0,
            "strict_meaningful_non_micro_winner_count": 0,
            "phase1_gate_open": False,
        },
    )
    _write(
        inventory,
        {
            "metrics": {
                "layout_prior_example_count": 44884,
                "region_heatmap_example_count": 10000,
                "candidate_quality_example_count": 146,
            }
        },
    )
    _write(schema, {"label_separation_rule": "training priors and candidate quality separate"})
    _write(quality, {"summary": {"official_like_improving": 2}})

    result = build_input_inventory(
        step7n,
        inventory,
        quality,
        quality,
        quality,
        tmp_path / "out.json",
        tmp_path / "guard.md",
        step7ml_g_schema_path=schema,
    )

    assert result["decision"] == "promote_to_training_demand_atlas"
    assert result["step7n_phase0"]["phase1_gate_open"] is False
    assert result["step7ml_g"]["layout_prior_example_count"] == 44884


def test_training_demand_atlas_uses_allowlist_and_omits_forbidden_terms(tmp_path) -> None:
    inventory = tmp_path / "inventory.json"
    schema = tmp_path / "schema.json"
    layout = tmp_path / "step7ml_g_layout_prior_examples.json"
    heatmap = tmp_path / "step7ml_g_region_heatmap_examples.json"
    _write(
        inventory,
        {"metrics": {"layout_prior_example_count": 1, "region_heatmap_example_count": 1}},
    )
    _write(schema, {"label_separation_rule": "separate"})
    _write(
        layout,
        {
            "rows": [
                {
                    "closure_type": "mib",
                    "block_count": 6,
                    "closure_area": 100.0,
                    "closure_aspect": 1.2,
                    "b2b_edge_count": 3,
                    "p2b_edge_count": 4,
                    "pin_count": 5,
                    "label_contract": "fp_sol should not be copied to atlas",
                }
            ]
        },
    )
    _write(
        heatmap,
        {
            "rows": [
                {
                    "region_distribution": [[0.7, 0.3], [0.0, 0.0]],
                    "area_distribution": [[0.4, 0.2], [0.0, 0.0]],
                    "free_space_proxy": [[0.9, 0.5], [0.0, 0.0]],
                }
            ]
        },
    )

    summary = build_training_demand_atlas(
        inventory,
        schema,
        layout,
        heatmap,
        [19, 24],
        tmp_path / "atlas.jsonl",
        tmp_path / "summary.json",
        tmp_path / "summary.md",
    )

    rows = [json.loads(line) for line in (tmp_path / "atlas.jsonl").read_text().splitlines()]
    assert summary["decision"] == "promote_to_prior_calibration"
    assert summary["represented_case_count"] == 2
    assert summary["forbidden_validation_label_term_count"] == 0
    assert summary["source_ledger"]["accepted_source_count"] == 5
    assert summary["rejected_sources"]
    assert all("fp_sol" not in json.dumps(row).lower() for row in rows)
    assert {row["case_id"] for row in rows} == {19, 24}
