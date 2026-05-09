#!/usr/bin/env python3
"""Run Step7Q-B constrained operator-source risk/ranking smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7q_operator_policy_smoke import run_operator_policy_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--branch-summary", type=Path, required=True)
    parser.add_argument("--scores-out", type=Path, required=True)
    parser.add_argument("--deck-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--request-count", type=int, default=96)
    parser.add_argument("--min-per-case", type=int, default=5)
    parser.add_argument("--max-case-share", type=float, default=0.25)
    args = parser.parse_args()
    summary = run_operator_policy_smoke(
        args.examples,
        args.branch_summary,
        args.scores_out,
        args.deck_out,
        args.summary_out,
        args.markdown_out,
        request_count=args.request_count,
        min_per_case=args.min_per_case,
        max_case_share=args.max_case_share,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "selected_request_count": summary["selected_request_count"],
                "represented_case_count": summary["represented_case_count"],
                "risk_profile_pass": summary["risk_profile_pass"],
                "strict_gate_pass": summary["strict_gate_pass"],
                "phase4_gate_open": summary["phase4_gate_open"],
                "allowed_next_phase": summary["allowed_next_phase"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
