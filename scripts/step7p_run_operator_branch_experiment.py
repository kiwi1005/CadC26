#!/usr/bin/env python3
"""Run Step7P causal-operator branch comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_operator_branch_experiment import (
    run_operator_branch_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--baseline-replay-summary", type=Path, required=True)
    parser.add_argument("--branch-rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summary = run_operator_branch_experiment(
        args.atlas,
        args.baseline_replay_summary,
        args.branch_rows_out,
        args.summary_out,
        args.markdown_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "best_branch_name": summary["best_branch_name"],
                "effective_partial_branch_count": summary["effective_partial_branch_count"],
                "phase4_open_branch_count": summary["phase4_open_branch_count"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
