from __future__ import annotations

import json

from puzzleplace.experiments.step7n_archive_lineage import (
    DEFAULT_MEANINGFUL_COST_EPS,
    extract_candidate_rows,
    mine_archive_lineage,
    normalize_row,
)


def _exact_row(
    *,
    case_id: int = 24,
    candidate_id: str = "case024:region:001",
    cost_delta: float = -2e-7,
    source_family: str = "regional_window",
    displacement: float = 0.02,
) -> dict[str, object]:
    return {
        "schema": "step7m_phase2_replay_row_v1",
        "case_id": str(case_id),
        "candidate_id": candidate_id,
        "source_family": source_family,
        "hard_feasible": True,
        "hard_feasible_nonnoop": True,
        "non_original_non_noop": True,
        "fresh_metric_available": True,
        "actual_objective_vector": {
            "hpwl_delta": -0.1,
            "bbox_area_delta": 0.0,
            "soft_constraint_delta": 0.0,
            "official_like_cost_delta": cost_delta,
        },
        "cost_components": {"displacement": displacement},
    }


def test_extract_candidate_rows_reads_rows_and_report_lists() -> None:
    payload = {
        "rows": [{"candidate_id": "a", "case_id": 1}],
        "fields": ["not", "rows"],
    }
    assert [row["candidate_id"] for row in extract_candidate_rows(payload)] == ["a"]

    report = {"official_like_winners": [{"candidate_id": "b", "case_id": 2}]}
    rows = extract_candidate_rows(report)
    assert rows[0]["candidate_id"] == "b"
    assert rows[0]["_source_list_key"] == "official_like_winners"


def test_normalize_row_requires_exact_comparable_for_strict_winner(tmp_path) -> None:
    exact = normalize_row(
        _exact_row(),
        tmp_path / "step7m_phase2_replay_rows.jsonl",
        row_index=0,
        meaningful_cost_eps=DEFAULT_MEANINGFUL_COST_EPS,
        non_micro_thresholds=(1e-4, 1e-3, 1e-2),
    )
    assert exact["metric_confidence"] == "exact_replay_comparable"
    assert exact["strict_archive_candidate"] is True
    assert exact["non_micro"] is True

    proxy = normalize_row(
        _exact_row(candidate_id="case024:sweep", source_family="sweep"),
        tmp_path / "step7n_h_e_sweep_results.json",
        row_index=0,
        meaningful_cost_eps=DEFAULT_MEANINGFUL_COST_EPS,
        non_micro_thresholds=(1e-4, 1e-3, 1e-2),
    )
    assert proxy["metric_confidence"] == "proxy_or_sweep_only"
    assert proxy["strict_archive_candidate"] is False


def test_mine_archive_lineage_reports_manifest_sources_and_concentration(tmp_path) -> None:
    exact_path = tmp_path / "step7m_phase2_replay_rows.jsonl"
    exact_path.write_text(
        "\n".join(
            [
                json.dumps(_exact_row(candidate_id="case024:region:001")),
                json.dumps(
                    _exact_row(
                        case_id=25,
                        candidate_id="case025:micro:001",
                        cost_delta=-1e-8,
                        source_family="micro_axis_corridor",
                        displacement=1e-6,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )
    proxy_path = tmp_path / "step7n_h_e_sweep_results.json"
    proxy_path.write_text(
        json.dumps({"rows": [_exact_row(candidate_id="case024:sweep:001")]}),
        encoding="utf-8",
    )
    self_output = tmp_path / "step7n_phase0_lineage_rows.jsonl"
    self_output.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.json"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(str(path) for path in (exact_path, proxy_path, self_output, missing)),
        encoding="utf-8",
    )

    summary = mine_archive_lineage(
        manifest,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        tmp_path / "ledger.json",
        tmp_path / "by_case.json",
        tmp_path / "taxonomy.json",
        tmp_path / "summary.md",
        meaningful_cost_eps=DEFAULT_MEANINGFUL_COST_EPS,
        non_micro_thresholds=(1e-4, 1e-3, 1e-2),
    )

    assert summary["decision"] == "diagnostic_only_due_concentration"
    assert summary["strict_archive_candidate_count"] == 1
    assert summary["strict_meaningful_non_micro_winner_count"] == 1
    assert summary["case024_share"] == 1.0
    assert summary["case025_share"] == 0.0
    assert summary["largest_case_id"] == "24"
    assert summary["metric_confidence_counts"]["proxy_or_sweep_only"] == 1
    assert summary["source_status_counts"]["skipped_missing"] == 1
    assert summary["source_status_counts"]["skipped_schema"] == 1
    assert summary["gnn_rl_gate_open"] is False
