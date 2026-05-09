#!/usr/bin/env python3
"""Run Step7Q-C constrained operator-parameter expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_parameter_expansion import run_operator_parameter_expansion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--selected-source-deck", type=Path, required=True)
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--candidates-out", type=Path, required=True)
    parser.add_argument("--deck-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=96)
    parser.add_argument("--min-per-case", type=int, default=5)
    parser.add_argument("--max-case-share", type=float, default=0.25)
    args = parser.parse_args()
    summary = run_operator_parameter_expansion(
        args.examples,
        args.selected_source_deck,
        args.policy_summary,
        args.candidates_out,
        args.deck_out,
        args.summary_out,
        args.markdown_out,
        target_count=args.target_count,
        min_per_case=args.min_per_case,
        max_case_share=args.max_case_share,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "candidate_count": summary["candidate_count"],
                "selected_expansion_count": summary["selected_expansion_count"],
                "represented_case_count": summary["represented_case_count"],
                "fresh_replay_required": summary["fresh_replay_required"],
                "phase4_gate_open": summary["phase4_gate_open"],
                "allowed_next_phase": summary["allowed_next_phase"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
