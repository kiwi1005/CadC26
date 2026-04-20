#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.rollout import beam_rollout, greedy_rollout
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases, run_bc_overfit


def main() -> None:
    case_limit = int(os.environ.get("ROLLOUT_CASE_LIMIT", "5"))
    max_traces = int(os.environ.get("ROLLOUT_MAX_TRACES", "2"))
    hidden_dim = int(os.environ.get("ROLLOUT_HIDDEN_DIM", "64"))
    epochs = int(os.environ.get("ROLLOUT_EPOCHS", "20"))

    cases = load_validation_cases(case_limit=case_limit)
    dataset = build_bc_dataset_from_cases(cases, max_traces_per_case=max_traces)
    policy, summary = run_bc_overfit(dataset, hidden_dim=hidden_dim, epochs=epochs, lr=1e-3, seed=0)

    results = []
    for case in cases:
        greedy = greedy_rollout(case, policy)
        beam = beam_rollout(case, policy, beam_width=4, per_state_candidates=3)
        results.append(
            {
                "case_id": case.case_id,
                "greedy": {
                    "placed_count": greedy.placed_count,
                    "all_blocks_placed": greedy.all_blocks_placed,
                    "stopped_reason": greedy.stopped_reason,
                    "feasible": greedy.feasible,
                },
                "beam": {
                    "placed_count": beam.placed_count,
                    "all_blocks_placed": beam.all_blocks_placed,
                    "stopped_reason": beam.stopped_reason,
                    "feasible": beam.feasible,
                },
            }
        )
    payload = {
        "bc_summary": {
            "initial_loss": summary.initial_loss,
            "final_loss": summary.final_loss,
            "primitive_accuracy": summary.primitive_accuracy,
            "block_accuracy": summary.block_accuracy,
        },
        "results": results,
    }
    output_path = ROOT / "artifacts" / "reports" / "agent9_rollout_validation0_4.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
