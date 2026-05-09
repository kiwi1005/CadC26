from __future__ import annotations

from pathlib import Path

from puzzleplace.repack.causal_closure_repacker import run_synthetic_repacker


def test_synthetic_causal_repacker_contract() -> None:
    summary = run_synthetic_repacker(
        Path("tests/fixtures/step7p"),
        Path("/tmp/step7p_phase2_report.json"),
        Path("/tmp/step7p_phase2_report.md"),
        Path("/tmp/step7p_phase2_contract.json"),
    )

    assert summary["decision"] == "promote_to_causal_request_deck"
    assert summary["fixture_count"] == 5
    assert summary["all_fixture_legal_non_noop"] is True
    assert summary["all_no_overlap_pass"] is True
    assert summary["all_area_preserved_pass"] is True
    assert summary["all_fixed_preplaced_unchanged_pass"] is True
    assert summary["mib_equal_shape_guard_pass"] is True
    assert summary["boundary_guard_pass"] is True
    assert summary["hpwl_only_soft_regression_rejected"] is True
    assert summary["fixtures_with_three_or_more_pareto"] >= 3
    assert summary["phase3_gate_open"] is True
    assert summary["gnn_rl_gate_open"] is False
