from __future__ import annotations

import json
from pathlib import Path


SUMMARY_PATH = Path("artifacts/research/step7r_phase0_stagnation_lock.json")

REQUIRED_KEYS = {
    "decision",
    "predecessor_step",
    "predecessor_strict_winner_count",
    "predecessor_soft_regression_rate",
    "predecessor_bbox_regression_rate",
    "predecessor_overlap_after_splice_count",
    "predecessor_actual_all_vector_nonregressing_count",
    "open_step",
    "gnn_rl_gate_open",
    "phase1_gate_open",
    "generated_at_utc",
}


def test_step7r_phase0_stagnation_lock_contract() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_KEYS <= summary.keys()
    assert summary["decision"] == "open_chain_move_operator"
    assert summary["predecessor_strict_winner_count"] == 0
    assert summary["phase1_gate_open"] is True
