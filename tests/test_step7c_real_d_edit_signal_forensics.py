from __future__ import annotations

from puzzleplace.diagnostics.edit_signal_forensics import (
    decision_for_step7c_real_d,
    failure_taxonomy,
    feature_delta_correlation,
    missing_field_report,
    slot_scoring_calibration,
    summary_counts,
    tradeoff_report,
    unified_candidate_table,
    winner_loser_examples,
)


def _row(
    strategy: str,
    *,
    official_delta: float,
    hpwl_delta: float,
    source: str = "b",
    feasible: bool = True,
    official_improving: bool = False,
    safe: bool = False,
    no_op: bool = False,
    route: str = "local",
    failure: str = "none",
    changed: int = 1,
    window: int | None = None,
) -> dict[str, object]:
    return {
        "case_id": 1,
        "candidate_id": f"case001:{strategy}:{source}",
        "strategy": strategy,
        "descriptor_locality_class": route,
        "actual_locality_class": route,
        "route_lane": "bounded_local_noop_repair" if route == "local" else f"{route}_lane",
        "report_only": route == "global",
        "after_route_official_like_hard_feasible": feasible,
        "official_eval_available": True,
        "no_op": no_op,
        "safe_improvement": safe,
        "official_like_cost_improving": official_improving,
        "changed_block_count": changed,
        "changed_block_fraction": changed / 10.0,
        "window_block_count": window,
        "macro_closure_size": 0,
        "affected_region_count": 1,
        "free_space_fit_ratio": 0.5,
        "internal_trial_count": 4,
        "internal_feasible_trial_count": 2,
        "internal_best_official_like_cost_delta": official_delta,
        "hpwl_delta": hpwl_delta,
        "bbox_area_delta": 0.0,
        "official_like_cost_delta": official_delta,
        "official_like_hpwl_gap_delta": hpwl_delta / 100.0,
        "official_like_area_gap_delta": 0.0,
        "official_like_violation_delta": 0.0,
        "mib_group_boundary_soft_delta": {
            "mib_delta": 0,
            "grouping_delta": 0,
            "boundary_delta": 0,
            "total_delta": 0,
        },
        "failure_attribution": failure,
        "slot_assignment": {"1": [1.0, 1.0]} if window else {},
    }


def _table() -> list[dict[str, object]]:
    b_rows = [
        _row(
            "hpwl_directed_local_nudge",
            official_delta=-0.01,
            hpwl_delta=-1.0,
            official_improving=True,
            safe=True,
        ),
        _row(
            "slack_fit_insertion",
            official_delta=-0.02,
            hpwl_delta=-1.2,
            official_improving=True,
            safe=True,
        ),
        _row("bbox_shrink_nudge", official_delta=0.03, hpwl_delta=0.5, failure="poor_targeting"),
    ]
    c_rows = [
        _row(
            "critical_net_slack_fit_window",
            official_delta=0.02,
            hpwl_delta=-0.2,
            source="c",
            safe=True,
            failure="metric_tradeoff_failure",
            window=4,
        ),
        _row(
            "legacy_step7g_global_report",
            official_delta=9.0,
            hpwl_delta=3.0,
            source="c",
            feasible=False,
            route="global",
            failure="global_report_only_not_repaired",
            changed=8,
            window=8,
        ),
    ]
    return unified_candidate_table(b_rows, c_rows)  # type: ignore[arg-type]


def test_unified_candidate_table_classifies_rows() -> None:
    table = _table()
    assert len(table) == 5
    winners = [row for row in table if row["official_like_improving"]]
    assert {row["strategy"] for row in winners} == {
        "hpwl_directed_local_nudge",
        "slack_fit_insertion",
    }
    assert any(row["candidate_class"] == "infeasible" for row in table)
    assert any(row["window_block_count"] == 4 for row in table)


def test_forensics_reports_required_measurements() -> None:
    table = _table()
    summaries = summary_counts(table)
    correlations = feature_delta_correlation(table)
    tradeoffs = tradeoff_report(table)
    slot = slot_scoring_calibration(table)
    failures = failure_taxonomy(table)
    examples = winner_loser_examples(table)
    missing = missing_field_report(table)

    assert summaries["candidate_count_by_source_step"] == {
        "step7c_real_b": 3,
        "step7c_real_c": 2,
    }
    assert summaries["official_like_improving_count_by_strategy"]["slack_fit_insertion"] == 1
    assert "hpwl_delta_vs_official_like_delta_summary" in correlations
    assert tradeoffs["overall"]["hpwl_gain_count"] >= 3
    assert slot["scored_candidate_count"] == len(table)
    assert failures["failure_family_counts"]["global_report_only"] == 1
    assert examples["top_positive_examples"][0]["strategy"] == "slack_fit_insertion"
    assert all(count == 0 for count in missing["required_missing_counts"].values())


def test_decision_prefers_expanding_clear_b_local_winners_when_c_underperforms() -> None:
    table = _table()
    summaries = summary_counts(table)
    correlations = feature_delta_correlation(table)
    failures = failure_taxonomy(table)
    missing = missing_field_report(table)
    assert decision_for_step7c_real_d(table, summaries, correlations, failures, missing) == (
        "expand_slack_fit_local_lane"
    )
