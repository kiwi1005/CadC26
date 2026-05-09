from __future__ import annotations

import json
from pathlib import Path

from puzzleplace.experiments.step7p_stagnation_lock import build_stagnation_lock


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stagnation_lock_starts_only_when_all_stop_evidence_is_present(tmp_path) -> None:
    step7l = _write(tmp_path / "l.json", {"decision": "complete_step7l"})
    step7m2 = _write(tmp_path / "m2.json", {"meaningful_official_like_improving_count": 0})
    step7m3 = _write(tmp_path / "m3.json", {"meaningful_official_like_improving_total": 0})
    step7m4 = _write(tmp_path / "m4.json", {"meaningful_official_like_improving_count": 0})
    step7n = _write(
        tmp_path / "n.json",
        {
            "decision": "stop_no_archive_signal",
            "strict_archive_candidate_count": 0,
            "strict_meaningful_non_micro_winner_count": 0,
            "phase1_gate_open": False,
        },
    )
    step7ml_i = _write(tmp_path / "i.json", {"official_like_improving_count": 2})
    step7ml_j = _write(
        tmp_path / "j.json",
        {"summary": {"official_like_improving_count": 2, "soft_regression_count": 114}},
    )
    step7ml_k = _write(
        tmp_path / "k.json",
        {"summary": {"official_like_improving_count": 2, "metric_regressing_count": 62}},
    )
    target = _write(tmp_path / "target.json", {"target_screening": {"baseline_winner_count": 3}})
    step7o = _write(
        tmp_path / "o.json",
        {
            "decision": "keep_prior_report_only",
            "phase3_gate_open": False,
            "concentration_pass": False,
            "step7ml_winner_baseline": 2,
        },
    )

    summary = build_stagnation_lock(
        step7l,
        step7m2,
        step7m3,
        step7m4,
        step7n,
        step7ml_i,
        step7ml_j,
        step7ml_k,
        target,
        step7o,
        tmp_path / "out.json",
        tmp_path / "out.md",
        tmp_path / "audit.jsonl",
    )

    assert summary["decision"] == "start_causal_closure_repack"
    assert summary["phase1_gate_open"] is True
    assert summary["step7ml_winner_baseline"] == 2
    assert summary["step7o_phase3_gate_open"] is False
    assert "gnn_rl_training" in summary["forbidden_next_steps"]
    audit_rows = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_rows) == 5
