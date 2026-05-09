#!/usr/bin/env python3
"""Check Step7P readiness for constrained RL/GNN operator learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7p_gnn_rl_readiness import evaluate_gnn_rl_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--branch-summary", type=Path, required=True)
    parser.add_argument("--blocker-diagnosis", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_gnn_rl_readiness(
        args.atlas,
        args.branch_summary,
        args.blocker_diagnosis,
        args.replay_summary,
        args.summary_out,
        args.markdown_out,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
                "allowed_phase": summary["allowed_phase"],
                "next_recommendation": summary["next_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
