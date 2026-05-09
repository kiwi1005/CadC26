from __future__ import annotations

from puzzleplace.experiments.step7m_corridor_ablation import (
    ablation_row,
    row_matches_ablation,
    summarize_ablation,
)


def _row(
    *,
    gate_mode: str = "wire_safe",
    hpwl_proxy: float = -1.0,
    hpwl_actual: float = -1.0,
    soft_actual: float = 0.0,
    metric_regressing: bool = False,
) -> dict[str, object]:
    return {
        "case_id": "19",
        "gate_mode": gate_mode,
        "source_family": "micro_axis_corridor",
        "hard_feasible_nonnoop": True,
        "metric_regressing": metric_regressing,
        "official_like_cost_delta": 0.1 if metric_regressing else -1e-8,
        "hpwl_delta": hpwl_actual,
        "bbox_area_delta": 0.0,
        "soft_constraint_delta": soft_actual,
        "quality_gate_status": "metric_tradeoff_report_only",
        "proxy_objective_vector": {
            "hpwl_delta_proxy": hpwl_proxy,
            "bbox_area_delta_proxy": 0.0,
            "boundary_delta_proxy": 0.0,
            "group_delta_proxy": 0.0,
            "mib_delta_proxy": 0.0,
        },
        "proxy_actual_signs": {
            "hpwl_nonregression_match": hpwl_proxy <= 0 and hpwl_actual <= 0,
            "bbox_nonregression_match": True,
            "soft_nonregression_match": soft_actual <= 0,
        },
    }


def test_row_matches_ablation_uses_proxy_vector_components() -> None:
    strict = _row()
    loose = _row(gate_mode="soft_repair_budgeted", hpwl_proxy=0.2)

    assert row_matches_ablation(strict, "hpwl_bbox_soft") is True
    assert row_matches_ablation(loose, "hpwl_bbox_soft") is False
    assert row_matches_ablation(loose, "bbox_soft") is True
    assert row_matches_ablation(loose, "soft_budgeted_gate") is True


def test_ablation_row_reports_regression_rates_and_sign_precision() -> None:
    rows = [_row(), _row(hpwl_actual=0.2, metric_regressing=True)]
    report = ablation_row("all_phase2", rows)

    assert report["request_count"] == 2
    assert report["metric_regressing_count"] == 1
    assert report["metric_regression_rate"] == 0.5
    assert report["actual_hpwl_regression_count"] == 1
    assert report["proxy_actual_sign_precision"]["component_total"] == 6


def test_summarize_ablation_recommends_strict_gate_when_it_improves_rate(tmp_path) -> None:
    all_row = ablation_row(
        "all_phase2", [_row(), _row(hpwl_proxy=0.2, hpwl_actual=0.2, metric_regressing=True)]
    )
    strict_row = ablation_row("hpwl_bbox_soft", [_row()])
    summary = summarize_ablation(
        [all_row, strict_row],
        replay_rows_path=tmp_path / "r.jsonl",
        rows_out_path=tmp_path / "a.jsonl",
    )

    assert summary["decision"] == "tighten_to_hpwl_bbox_soft_then_multiblock_v1"
    assert summary["best_family"] == "hpwl_bbox_soft"
    assert summary["gnn_rl_gate_open"] is False
